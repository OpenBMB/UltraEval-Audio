"""Judge prompts for EmergentTTS-Eval.

Copied verbatim (including the original leading indentation) from the
official benchmark's ``prompts.py``
(https://github.com/boson-ai/EmergentTTS-Eval-public, Apache-2.0), so the
model-as-a-judge setup stays byte-compatible with the published leaderboard
and with ``score_existing_results.py``.  The indentation inside these strings
is part of the prompt and must not be stripped.

Trailing spaces are written as ``\x20`` escapes because the prompt bytes
include them; a pre-commit trailing-whitespace hook would otherwise strip
them and silently change the judge prompt.

Placeholders use the official triple-brace markers and are filled with
``str.replace`` (the texts can contain braces, e.g. equations).
"""

SYSTEM_PROMPT_JUDGER = """You are an AI audio assistant acting as a strong reward model for evaluating two TTS(text-to-speech) systems by carefully analyzing their generated speech for intonation, prosody, pronunciation, expressiveness, etc."""

USER_MESSAGE_WIN_RATE = """Your goal is to judge two TTS(text-to-speech) systems and analyze which system synthesizes speech corresponding to a particular text better than the other one and determine the winner based on the scoring criterion.
    You will rate each system a score between 0 and 3 based on how well it synthesizes speech corresponding to a particular text called **text_to_synthesize**, then do their comparative analysis and provide your final judgement.
    A good system will generate speech that sounds realistic and human-like, and it captures the specific nuances of the text.

    You will be provided with the **text_to_synthesize** which is the text both TTS systems had to synthesize,\x20
    the **text_category** and the **evaluation_criterion** corresponding to the **text_category**, in which you will be made aware of the **evaluation dimension** you will focus on, and the **scoring criteria** you will use to score the TTS systems.
    You will also be provided with the **output_format**, which dictates the format of the output you need to follow as a judger.
    Finally, you will be provided with the synthesized speech from the TTS system 1 **synthesized_speech_1** and then from TTS system 2 **synthesized_speech_2**.

    **text_to_synthesize**
    {{{text_to_synthesize}}}


    **text_category**
    {{{text_category}}}


    **evaluation_criterion**
    {{{evaluation_criterion}}}

    NOTE: If the generated speech is very poor and does not synthesise the text correctly, you will provide a score of 0 to that TTS system.
    GLOBAL CONSIDERATIONS(**VERY IMPORTANT FOR COMPARISON**):\x20
        - It is imperative to compare the two systems **ONLY** on the basis of the **evaluation_dimension**, that means, you **WILL NOT** let the following types of **BIASES** affect your judgement:\x20
            - The acoustical quality of the audio, background noise or clarity.
            - The gender and timbre features of the speaker.
            - Any other factors that are not related to the **evaluation_dimension**.
            - Systems demonstrating exaggerated expressiveness should not be rewarded more **UNLESS** those features are relevant to the **evaluation_dimension**.
        - Tie-break procedure
            1. If the overall score_1 and score_2 are equal, use this protocol.
            2. For the chosen **evaluation_dimension**, inspect every comparable component:
                • Note similarities.\x20\x20
                • Note differences and label each as:
                    - Subtle: hardly noticeable to a typical human listener.\x20\x20
                    - Significant: clearly influences human perception.
            3. Count the significant differences that benefit each system.
            4. Decision:
                - No significant differences, or counts are equal → declare a tie.
                - Otherwise → declare the system with the higher count of significant advantages the winner.

    **output_format**
    You will output a json dictionary as follows:
    ```json
    {
    "reasoning_system_1": str = Reasoning chain based on the **Reasoning guidelines:** for the synthesized speech from TTS system 1.
    "reasoning_system_2": str = Reasoning chain based on the **Reasoning guidelines:** for the synthesized speech from TTS system 2, **INDEPENDENT** of the performance of TTS system 1.
    "system_comparison": str = Keeping in mind the GLOBAL CONSIDERATIONS, compare and contrast the two systems based on your output in reasoning_system_1 and reasoning_system_2 and also by analyzing both audios again. Provide very fine-grained reasoning for which system won, or if the comparison results in an even tie.
    "score_1": int = Your score for the synthesized speech from TTS system 1 between 0 and 3, based on the **evaluation_criterion** and what you have mentioned in reasoning_system_1.
    "score_2": int = Your score for the synthesized speech from TTS system 2 between 0 and 3, based on the **evaluation_criterion** and what you have mentioned in reasoning_system_2.
    "winner": int = The winner of the comparison between TTS system 1 and TTS system 2. Output 1 if TTS system 1 wins, 2 if TTS system 2 wins, output 0 if this will be considered as an even tie.
    }
    - Note: Ensure the json structure is followed and the json output **MUST** be parsable without errors.(For example, escape the quotes whereever you add them inside a field of the json, all brackets and braces should be correctly paired.)

    Now you will be provided with the synthesized speech from the TTS system 1, please analyze it carefully.

    **synthesized_speech_1**
\x20\x20\x20\x20"""

POST_AUDIO_1_MESSAGE = """
    Now you will be provided with the synthesized speech from the TTS system 2, please analyze it carefully. After that provide the judgment following the **output_format** ensuring parsability.
    **synthesized_speech_2**
\x20\x20\x20\x20"""

CATEGORY_TO_CRITERION_MAP = {
    "Questions": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate the ability of the TTS system to apply correct intonation patterns: Interrogative for questions, declarative for statements, etc.
            - Questions usually have a distinct pitch movement, often rising at the end in yes/no questions, while wh-questions may have a more neutral or falling tone.
            - Statements between questions should have an intonation pattern that differentiates them from the questions and makes it clear that it is a statement.
            - You have to be careful that texts can have multiple correct intonation patterns, so place appropriate weight on parts where intonation is not very clear.

        **Example:** "Did you see the message? Well I hope you did. But please tell me that you actually did?"
        **Explanation:**\x20
            - There maybe multiple correct patterns to render this speech with, but we want to judge if the TTS system has made an attemp at correctly conveying the interrogative intonation for the 2 questions, and the declarative intonation for the statement between the questions.

        **Note:**
            - The **text_to_synthesize** may contain multiple questions with or without the question mark, you have to correctly differentiate between the questions and the statements.

        **Rating Scale:**
        1: All intonation patterns incorrect
        2: Some intonation patterns are largely correct but some are incorrect
        3: All intonation patterns are correct and convey the question nature perfectly
\x20\x20\x20\x20\x20\x20\x20\x20
        **Reasoning guidelines:**
            1. Mention which parts need to be rendered with interrogative intonation and which with declarative intonation.
            2. Carefully list the crucial parts of the speech, the pertinent syllables and their precise timestamps.
            3. Analyze the audio multiple times to capture the intonation patters at the crucial parts.
            4. Finally, reason deeply and justify how the TTS system has performed and applied the intonation patterns at the crucial parts, and then what the final score for the TTS system should be.
\x20\x20\x20\x20\x20\x20\x20\x20""",
    "Emotions": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate the ability of the TTS system to express emotions naturally, using variations in pitch, loudness, rhythm, etc. and demonstrate tone variations between the quoted dialogues and the narrative text.
            - The TTS system has to generate speech as if it is narrating the **text_to_synthesize**, which means showing natural and strong emotional expressiveness for the quoted dialogues.

        **Example:** "Full of joy, he exclaimed: "I can't believe it! This is amazing!". But then, a sudden realization dawned on him and he said "Okay okay wait wait, I think this may not be such a good idea after all."
        **Explanation:**\x20
            - The text inside the first quotes "I can't believe it! This is amazing!" should sound excited and joyful, not robotic.
            - The text inside the second quotes "Okay okay wait wait, I think this may not be such a good idea after all." should sound disappointed and frustrated and this contrasting emotion should be clearly noticeable.\x20
            - The narrative between/around the quotes should be distinct than the dialogues and should be spoken with the appropriate narrative tone.

        **Rating Scale:**
        1: Fails to express emotions in the quoted dialogues, and the transition between the quoted dialogues and the narrative is flat and not distinct.
        2: Synthesises some quoted dialogues with emotions but fails to synthesise others, OR, the rendered emotions are not very natural and emphatic, OR, the tone bridging quoted dialogues and the narrative text cannot be distinguised/is barely discernible.
        3: Synthesises all quoted dialogues with natural and emphatic emotions, and the tone bridging quoted dialogues and the narrative text is clearly distinguishable.
\x20\x20\x20\x20\x20\x20\x20\x20
        **Note:**
            - The **text_to_synthesize** will not explicitly state the emotion for the quoted dialogues, you have to infer that from the context.

        **Reasoning guidelines:**
            1. Identify the emotional state in which the all the quoted dialogues should be spoken based on the context, identify the intensifying and contrasting emotions.
            2. Provide precise timestamps of **EVERY** crucial part of the quoted dialogue, and comment on the emotional expressiveness of these parts that are imporant to convey the **OVERALL** emotional tone of the dialogue.
            3. Analyze the boundry points, where quoted dialogues and narrative context meet, and provide precise timestamps of these parts, while reasoning how there may be a change in the emotional tone of the speech at these points.
            4. Finally, reason deeply and justify how the expressive the TTS system is, and how it has narrated the **text_to_synthesize**, and then what the final score for the TTS system should be.
\x20\x20\x20\x20\x20\x20\x20\x20""",
    "Syntactic Complexity": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate the ability of the TTS system to use **prosody (pausing, phrasing, intonation, stress)** to make complex sentence structures easily understandable.
            - It tests if the TTS can convey a syntactically very complex sentence such that it's meaning to the listener is clear and understandable, that is the main goal.
            - Occasionaly, the text may contain homographic words, in that case, the TTS system should pronounce the homographic words with appropriate pronunciation.

        **Example:** "The book that the professor who won the award wrote is on the table."
        **Explanation:**\x20
            - Without proper phrasing and intonation, it's hard to follow who did what or to identify the main subject ("the book") and the verb ("is").
            - The rest of the sentence—"that the professor who won the award wrote"—is a complex noun modifier (a series of nested relative clauses) describing "the book."
            - The core structure of the sentence is: "The book is on the table."
            - A TTS system must use appropriate prosody—pausing, stress, and intonation—to guide the listener naturally through the structure, signaling the main subject, distinguishing the embedded clauses, and connecting all parts coherently.

        **Note:**
            - This category is all about adding appropriate pauses, stress, and intonation, in absence of punctuation marks, **AND** in their presence too. We want to check if the indended meaning is conveyed correctly and that is all that matters.

        **Rating Scale:**
        1: The prosody makes the sentence structure confusing or leads to an incorrect meaning.
        2: The intended structure is mostly understandable, but the prosody (pauses, intonation, stress) sounds unnatural or confusing at some parts.
        3: The prosody correctly conveys the sentence structure, making the complex grammar easy to follow and clarifying the intended meaning of the sentence very clearly.
\x20\x20\x20\x20\x20\x20\x20\x20
        **Reasoning guidelines:**
        1. Elaborate the intended meaning of the sentence and untangle the complex syntax.
        2. Identify the syntactically complex parts of the speech that require appropriate prosody (pausing, phrasing, intonation, stress) to be understandable, also identify any homographs and their intended pronunciation, finally list all these crucial parts.
        3. Carefully analyze and provide precise timestamps of crucial prosodic features - pauses between phrases, changes in intonation, and stress patterns - that help clarify the sentence structure for each of the crucial parts.
        4. Evaluate how well the prosody helps to distinguish the meaning at these crucial parts, for example, distinguish between main clauses and subordinate clauses, avoid garden path effects, and other syntactic complexities(including homographs) identified in 2.
        5. Finally, reason deeply and justify how effectively the TTS system's prosodic features (or lack thereof) contribute to the comprehensibility of the **OVERALL**complex syntax, and then determine the final score for the TTS system.
\x20\x20\x20\x20\x20\x20\x20\x20""",
    "Foreign Words": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate the ability of the TTS system to correctly pronounce foreign words and phrases, either using their original pronunciation or a widely accepted anglicized version.
            - The goal for the system is to sound like a fluent bi-lingual speaker, seamlessly doing code-switching between the languages.

        **Example:** "During his shaadi, manoj went pura paagal and started dancing jaise ki wo ek actor hai."
        **Explanation:**\x20
            - The words "shaadi", "paagal", "jaise" and "actor" should be pronounced with an acceptable hindi pronunciation(as there is no anglicized version for these words).
            - The flow when switching between the two languages should be seamless and natural, without awkward pauses or jumps.

        **Note:**
            - Not all foreign words have an anglicized version, in that case the words should be pronounced with an acceptable pronunciation in that foreign language.

        **Rating Scale:**
        1: Pronounces the foreign words and phrases completely incorrectly.
        2: Applies foreign pronunciation but not entirely correctly, some words are pronounced correct but others are not and the natural flow during code-switching is disrupted.
        3: Correct rendering in the intended language or acceptable anglicized version for all words and phrases, and the natural flow during code-switching is maintained.
\x20\x20\x20\x20\x20\x20\x20\x20
        **Reasoning guidelines:**
            1. Identify the foreign words and phrases, and the language they belong to.
            2. Provide precise timestamps for **ALL** the foreign words and phrases in the speech.
            3. Analyze the audio multiple times, and provide a comment on the pronunciation of the foreign words, and if the system has gotten none, some or all of them correct.
            4. Finally, reason deeply and justify how the TTS system has performed based on pronunciation **AND** the flow at code-switching points, and then what the final score for the TTS system should be.
\x20\x20\x20\x20\x20\x20\x20\x20""",
    "Paralinguistics": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate how well the TTS system synthesis speech corresponding to paralinguistic cues present in the text. There can be multiple types of paralinguistic cues present in the text, like:
                - Interjections ("Hmmm", "Ooops").
                - Vocal sounds/onomatopoeia("Shhh!", "Achoo!", "Meow")
                - Emphasis using CAPS("He didn't REALLY mean it" has a different sound than "He didn't really MEAN it"), vowel elongation("Heyyyyyyy, okayyyyyyy"), hyphenation/syllable stress("ab-so-lutely", "im-por-tant"), etc.
                - Pacing cues(ellipses, punctuation(for example STOP.RIGHT.THERE)).
                - Stuttering and hesitation("I-I-I", "W-we-well...", etc.)
            - The TTS system has to correctly identify all the paralinguistic cues present in the text and render them how human speech would render them.

        **Example:** `"Ugh! I-I told you... DO NOT touch that! Seriously?!"`
        **Explanation:** The TTS should render the frustration ("Ugh!"), hesitation ("I-I", "..."), emphasis ("DO NOT"), and final incredulous annoyance ("Seriously?!") suggested by the text, not just read the words flatly.

        **Note:**
            - It is **VERY IMPORATANT** to recognize that we are looking for a plausible rendering of the paralinguistic cue, as a human would render them while speaking the text.
            - Paralinguistic realism is also affected by the emotional tone that cue represents, you will only focus on the emotional affect for the cues, not the emotional tone forother parts of the speech.

        **Rating Scale:**
        1: Fails to render the intended vocal effect(s); sounds neutral or wrong.
        2: Intention to render the vocal effect(s), but the delivery sounds unnatural, awkward, or inaccurate.
        3: Successfully and naturally produces the vocal effect(s) implied by the textual cues.
\x20\x20\x20\x20\x20\x20\x20\x20
        **Reasoning guidelines:**
            1. Identify and list all of the paralinguistic cues present in the text, and the plausible intended vocal effect for each of them.
            2. Provide precise timestamps for **ALL** the paralinguistic cues in the speech.
            3. Give detailed analysis for **ALL** cues by analyzing the audio multiple times, like how they are synthesized, if they match the intended vocal effect, and how realistic to human speech they are.
            4. Finally, reason deeply and justify how the TTS system has performed in rendering the paralinguistic cues, and then what the final score for the TTS system should be.
\x20\x20\x20\x20\x20\x20\x20\x20""",
    "Pronunciation": """
        **Evaluation Dimension:**\x20
            - In this category, we want to evaluate how well the TTS system pronounces non-trival words, numerals and special characters present in the text.
            - To be specific, this category includes **text_to_synthesize** that fits in ONE of the following complex pronunciation categories and the TTS system has to render the **text_to_synthesize** correctly:
                1. Text with currency and numerals in different formats\x20
                2. Text with dates and time-stamps in different formats\x20
                3. Texts with email addressess, passwords and urls in different formats.
                4. Texts with complex street addresses or location references.\x20
                5. Texts with equations and notations from the STEM field.\x20
                6. Texts that have **BOTH** an initialism(pronounced initial by initial) and acronym(pronounced as a whole word).
                7. Texts with repeated tounge twisters.

        **Example:** "The equation e^(i*pi) + 1 = 0 is a famous equation in mathematics."
        **Explanation:**\x20
            - The equation "e^(i*pi) + 1 = 0" should be pronounced with the appropriate pronunciation, like "The equation e to the power of i times pi plus 1 equals 0 is a famous equation in mathematics."

        **Note:**
            - It is crucial to understand what the most natural pronunciation of the given text will be, sometimes it maybe helpful to think in reverse, i.e, if **text_to_synthesize** is actually the transcription of an audio, what would that audio sound like? The TTS system should synthesize audio similar to that.
            - It is more ideal for a system to speak tounge twisters faster **WHILE** still maintaining complete clarity **AND** consistency in pronunciation.
            - Initialisms should be pronounced initially by initial(for example, FBI), and acronyms(for example, NASA) should be pronounced as a single word.
            - Case-sensitivity sometimes matters(for example, passwords, URL paths after the domain name, etc.), so make sure to recognize any case-sensitive parts and reward/penalize accordingly.

        **Rating Scale:**
        1: Incorrect synthesis of the critical parts, with missing or completely incorrect/inappropriate pronunciation.
        2. Partially correct pronunciation of the **SOME** of the critical parts.
        3. Completely correct pronunciation of **ALL** the critical parts.

        **Reasoning guidelines:**
            1. Identify the critical parts of the text that require correct pronunciation.
            2. Provide precise timestamps for **ALL** the critical parts in the speech, and the ideal pronunciation for the same.
            3. Give detailed analysis for **ALL** the critical parts by analyzing the audio multiple times, explain how they are synthesized, if they match the intended pronunciation, and how realistic to human speech they are.
            4. Finally, reason deeply and justify how the TTS system has performed in pronouncing the critical parts, and then what the final score for the TTS system should be.
\x20\x20\x20\x20\x20\x20\x20\x20""",
}


def get_win_rate_prompts(text_to_synthesize: str, category: str):
    """Build the (system, user, post-audio-1) judge prompts for one sample."""
    if category not in CATEGORY_TO_CRITERION_MAP:
        raise ValueError(
            "Unknown EmergentTTS-Eval category {!r}; expected one of {}".format(
                category, sorted(CATEGORY_TO_CRITERION_MAP)
            )
        )
    evaluation_criterion = CATEGORY_TO_CRITERION_MAP[category]
    user_message = (
        USER_MESSAGE_WIN_RATE.replace("{{{text_to_synthesize}}}", text_to_synthesize)
        .replace("{{{text_category}}}", category)
        .replace("{{{evaluation_criterion}}}", evaluation_criterion)
    )
    return SYSTEM_PROMPT_JUDGER, user_message, POST_AUDIO_1_MESSAGE
