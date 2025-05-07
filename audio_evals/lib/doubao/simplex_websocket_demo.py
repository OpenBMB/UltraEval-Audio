import json
import time
import uuid
import requests


def submit_task(appid, token, file_url, audio_info):

    submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"

    task_id = str(uuid.uuid4())

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1",
    }

    endfix = file_url.split(".")[-1]

    request = {
        "user": {"uid": "fake_uid"},
        "audio": {"url": file_url, "format": endfix, **audio_info},
        "request": {
            "model_name": "bigmodel",
            # "enable_itn": True,
            # "enable_punc": True,
            # "enable_ddc": True,
            "show_utterances": True,
            # "enable_channel_split": True,
            # "vad_segment": True,
            # "enable_speaker_info": True,
            "corpus": {
                # "boosting_table_name": "test",
                "correct_table_name": "",
                "context": "",
            },
        },
    }
    print(f"Submit task id: {task_id}")
    response = requests.post(submit_url, data=json.dumps(request), headers=headers)
    if (
        "X-Api-Status-Code" in response.headers
        and response.headers["X-Api-Status-Code"] == "20000000"
    ):
        print(
            f'Submit task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}'
        )
        print(
            f'Submit task response header X-Api-Message: {response.headers["X-Api-Message"]}'
        )
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        print(
            f'Submit task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n'
        )
        return task_id, x_tt_logid
    else:
        print(f"Submit task failed and the response headers are: {response.headers}")
        exit(1)
    return task_id


def query_task(task_id, x_tt_logid):
    query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid,  # 固定传递 x-tt-logid
    }

    response = requests.post(query_url, json.dumps({}), headers=headers)

    if "X-Api-Status-Code" in response.headers:
        print(
            f'Query task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}'
        )
        print(
            f'Query task response header X-Api-Message: {response.headers["X-Api-Message"]}'
        )
        print(
            f'Query task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n'
        )
    else:
        print(f"Query task failed and the response headers are: {response.headers}")
        exit(1)
    return response


def main():
    task_id, x_tt_logid = submit_task()
    while True:
        query_response = query_task(task_id, x_tt_logid)
        code = query_response.headers.get("X-Api-Status-Code", "")
        if code == "20000000":  # task finished
            print(query_response.json())
            print("SUCCESS!")
            exit(0)
        elif code != "20000001" and code != "20000002":  # task failed
            print("FAILED!")
            exit(1)
        time.sleep(1)


# 需要使用在线url，推荐使用TOS
file_url = ""
# 填入控制台获取的app id和access token
appid = ""
token = ""

if __name__ == "__main__":
    main()
