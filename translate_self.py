import os
import pandas as pd
from openai import OpenAI
from typing import Optional, List, Tuple
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import requests


def translate_text(
    text: str, src_lang: str, tgt_lang: str, api_key: Optional[str] = None
) -> str:
    """
    使用新的API端点将文本从源语言翻译为目标语言

    Args:
        text (str): 源文本
        src_lang (str): 源语言
        tgt_lang (str): 目标语言
        api_key (str, optional): 保留参数以保持接口兼容性，但不再使用

    Returns:
        str: 翻译后的文本
    """
    try:
        # 语言代码映射
        lang_mapping = {"zh": "zh-cn", "zh-cn": "zh-cn", "ko": "ko", "en": "en"}

        # 获取标准化的语言代码
        source_lang = lang_mapping.get(src_lang.lower(), src_lang.lower())
        target_lang = lang_mapping.get(tgt_lang.lower(), tgt_lang.lower())

        # 准备请求数据
        payload = {"language_pair": [source_lang, target_lang], "query": text}

        # 发送POST请求到新的API端点
        response = requests.post(
            "https://zapz.ali-dev.modelbest.cn/api/v1/car/language",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        # 检查响应状态
        response.raise_for_status()

        # 解析响应
        result = response.json()

        # 检查API响应状态
        if result.get("code") == 0 and result.get("data"):
            translated_text = result["data"].strip()
            return translated_text
        else:
            print(f"API返回错误: {result.get('message', 'Unknown error')}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"网络请求失败: {str(e)}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}")
        return ""
    except Exception as e:
        print(f"翻译失败: {str(e)}")
        return ""


def translate_batch(
    batch_data: List[Tuple[int, str, str, str]]
) -> List[Tuple[int, str, str, str]]:
    """
    批量翻译函数，用于多线程处理

    Args:
        batch_data: 包含 (index, text, src_lang, target_lang) 的列表

    Returns:
        包含 (index, target_lang, translated_text) 的列表
    """
    results = []
    for index, text, src_lang, target_lang in batch_data:
        try:
            translated = translate_text(text, src_lang, target_lang)
            results.append((index, target_lang, translated))
        except Exception as e:
            print(f"翻译第 {index} 条到 {target_lang} 失败: {str(e)}")
            results.append((index, target_lang, ""))
    return results


def translate_with_realtime_save(f_name, src_lang=None, max_workers=5, batch_size=10):
    """
    多线程翻译并实时保存结果

    Args:
        f_name (str): 输入文件路径
        src_lang (str): 源语言，如果不提供则从文件名推断
        max_workers (int): 最大线程数，默认5个
        batch_size (int): 批处理大小，默认10条
    """
    # 读取数据
    df = pd.read_json(f_name, lines=True)
    df = df.fillna("")

    # 如果没有指定源语言，从文件名推断
    if src_lang is None:
        src_lang = f_name.split("/")[-1].split(".")[0]

    # 创建输出文件名
    output_file = f_name.replace(".jsonl", "_self_translated.jsonl")

    # 检查是否已有输出文件，如果有则加载
    existing_df = None
    if os.path.exists(output_file):
        try:
            print(f"发现已有输出文件: {output_file}")
            existing_df = pd.read_json(output_file, lines=True)
            existing_df = existing_df.fillna("")
            print(f"已加载 {len(existing_df)} 条已翻译数据")

            for item in existing_df["text"].tolist():
                if item not in df["text"].tolist():
                    print(item)

            # 检查数据完整性，如果行数不匹配则重新开始
            if len(existing_df) != len(df):
                print(
                    f"警告：已有文件行数({len(existing_df)})与输入文件行数({len(df)})不匹配，将重新开始翻译"
                )
                existing_df = None
            else:
                # 将已有的翻译结果合并到当前DataFrame

                for index, row in existing_df.iterrows():
                    for target_lang in ["zh", "ko", "en"]:
                        if (
                            target_lang in row
                            and pd.notna(row[target_lang])
                            and row[target_lang] != ""
                        ):
                            df.at[index, target_lang] = row[target_lang]

                print(f"已合并已有的翻译结果, {len(df)} 条")

        except Exception as e:
            print(f"加载已有文件失败: {str(e)}，将重新开始翻译")
            existing_df = None

    print(f"开始多线程翻译，源语言: {src_lang}")
    print(f"总数据量: {len(df)} 条")
    print(f"线程数: {max_workers}, 批处理大小: {batch_size}")

    # 准备需要翻译的任务（只包含未翻译的）
    translation_tasks = []
    skipped_count = 0

    for index, row in df.iterrows():
        for target_lang in ["zh", "ko", "en"]:
            if target_lang in row and (
                pd.isna(row[target_lang]) or row[target_lang] == ""
            ):
                if pd.notna(row["text"]) and row["text"] != "":
                    translation_tasks.append(
                        (index, row["text"], src_lang, target_lang)
                    )
                else:
                    skipped_count += 1

    print(f"需要翻译的任务数: {len(translation_tasks)}")
    # 如果没有需要翻译的任务，直接保存并返回
    if len(translation_tasks) == 0:
        print("所有内容都已翻译完成！")
        df.to_json(output_file, orient="records", lines=True, force_ascii=False)
        df.to_excel(output_file.replace(".jsonl", ".xlsx"), index=False)
        return df

    # 分批处理
    total_batches = (len(translation_tasks) + batch_size - 1) // batch_size
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有批次的任务
        future_to_batch = {}
        for i in range(0, len(translation_tasks), batch_size):
            batch = translation_tasks[i : i + batch_size]
            future = executor.submit(translate_batch, batch)
            future_to_batch[future] = i // batch_size

        # 处理完成的任务
        with tqdm(total=len(translation_tasks), desc="翻译进度") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()

                    # 更新DataFrame
                    for index, target_lang, translated_text in results:
                        df.at[index, target_lang] = translated_text
                        processed_count += 1

                    # 更新进度条
                    pbar.update(len(results))

                    # 实时保存当前进度
                    if processed_count % (batch_size * 2) == 0:  # 每处理2个批次保存一次
                        print(f"\n保存进度: {processed_count}/{len(translation_tasks)}")
                        # 保存为JSONL格式
                        df.to_json(
                            output_file, orient="records", lines=True, force_ascii=False
                        )
                        # 同时保存为Excel格式作为备份
                        excel_output = output_file.replace(".jsonl", ".xlsx")
                        df.to_excel(excel_output, index=False)

                except Exception as e:
                    print(f"批次 {batch_idx} 处理失败: {str(e)}")

    # 最终保存
    print(f"\n翻译完成！保存结果到: {output_file}")
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    df.to_excel(output_file.replace(".jsonl", ".xlsx"), index=False)

    return df


def translate_2_zh():
    """
    翻译为中文的便捷函数
    """
    # 这里需要指定具体的文件路径
    files_to_translate = ["raw/zh.jsonl", "raw/ko.jsonl", "raw/en.jsonl"]

    for file_path in files_to_translate:
        if os.path.exists(file_path):
            print(f"\n开始处理文件: {file_path}")
            # 使用多线程版本，可以根据需要调整参数
            translate_with_realtime_save(file_path, max_workers=16, batch_size=10)
        else:
            print(f"文件不存在: {file_path}")


# 示例用法
if __name__ == "__main__":
    translate_2_zh()
