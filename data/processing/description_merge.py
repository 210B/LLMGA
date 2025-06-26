from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

character_data = [
    {
        "character": "Ophelia",
        "descriptions": [
            "Polonius\u2019s daughter, a beautiful young woman with whom Hamlet has been in love. Ophelia is a sweet and innocent young girl, who obeys her father and her brother, Laertes. Dependent on men to tell her how to behave, she gives in to Polonius\u2019s schemes to spy on Hamlet. Even in her lapse into madness and death, she remains maidenly, singing songs about flowers and finally drowning in the river amid the flower garlands she had gathered.",
            "The daughter of Polonius and sister of Laertes. Ophelia has received several tributes of love from Hamlet but rejects him after her father orders her to do so.  In general, Ophelia is controlled by the men in her life, moved around like a pawn in their scheme to discover Hamlet's distemper.  Moreover, Ophelia is regularly mocked by Hamlet and lectured by her father and brother about her sexuality.  She goes mad after Hamlet murders Polonius.  She later drowns.",
            "Polonius\u2019s daughter, Laertes\u2019 sister, and Hamlet\u2019s lover. Along with Gertrude, Ophelia is the only other female character in the play, Ophelia\u2019s actions and trajectory are unfortunately defined by the men around her. At the start of the play, Ophelia\u2014who has been in a relationship of undetermined seriousness with Hamlet for an unspecified amount of time\u2014is used as a pawn in her father Polonius\u2019s attempt to help Claudius and Gertrude ascertain the source of Hamlet\u2019s madness. Polonius believes a burgeoning rift between Ophelia and Hamlet is the cause\u2014in reality, Hamlet is, unbeknownst to the others at Elsinore, affecting madness in order to seem less suspicious or threatening as he investigates his father\u2019s murder. Ophelia, however, is ignorant of Hamlet\u2019s plan\u2014and as she interacts with him in service of her father\u2019s plot, Hamlet becomes so hurt by her transparent betrayal that he begins acting like even more of a lunatic towards Ophelia, cruelly suggesting she become a nun and making lewd sexual remarks towards her at every available opportunity. Between Hamlet\u2019s abuses\u2014and his murder of Polonius\u2014Ophelia eventually loses her own mind, succumbing fully to the madness at which Hamlet has only been playing. Ophelia eventually commits suicide, and though Hamlet claims to grieve her, no one\u2014least of all Ophelia\u2019s furious brother, Laertes\u2014believes him. A tragic figure whose life and death alike are coopted by the men around her, Ophelia is nonetheless able to do the one thing Hamlet, for all his musings on his desire to take his own life, is never able to do: she kills herself, reclaiming through a tragic action the only measure of agency she\u2019s had over her own life for as long as she\u2019s lived it.",
            "Daughter of Polonius, sister of Laertes,  is beloved of Hamlet."
        ]
    },
    {
        "character": "Claudius",
        "descriptions": [
            "The King of Denmark, Hamlet\u2019s uncle, and the play\u2019s antagonist. The villain of the play, Claudius is a calculating, ambitious politician, driven by his sexual appetites and his lust for power, but he occasionally shows signs of guilt and human feeling\u2014his love for Gertrude, for instance, seems sincere.",
            "Old Hamlet's brother, Hamlet's uncle, and Gertrude's newlywed husband.  He murdered his brother in order to seize the throne and subsequently married Gertrude, his erstwhile sister-in-law.  Claudius appears to be a rather dull man who is fond of the pleasures of the flesh, sex and drinking.  Only as the play goes on do we become certain that he is indeed guilty of murder and usurpation.  Claudius is the only character aside from Hamlet to have a soliloquy in the play.  When he is convinced that Hamlet has found him out, Claudius eventually schemes to have his nephew-cum-son murdered.",
            "Hamlet\u2019s uncle and stepfather, and the new King of Denmark. After the death of Hamlet\u2019s father and Claudius\u2019s brother, the former king, Claudius married his brother\u2019s wife, Gertrude, and assumed the throne of Denmark\u2014much to Hamlet\u2019s chagrin and suspicion. Early on in the play, the ghost of Hamlet\u2019s father appears to Hamlet and tells him that Claudius murdered him for the throne, then charges Hamlet with avenging the murder. Hamlet\u2019s hatred of Claudius intensifies after speaking with the ghost\u2014but still, Hamlet is unable to take the action needed to get revenge for his father. Claudius, meanwhile, interprets Hamlet\u2019s suspicion and anger as madness, and endeavors to find out the cause by recruiting Hamlet\u2019s friends Rosencrantz and Guildenstern. Claudius fears that Hamlet is onto him, and when Hamlet arranges a performance of a play whose events mirror the manner of King Hamlet\u2019s murder, Claudius at last confesses, in a lengthy soliloquy, to his dark, dastardly deeds. Alone on stage, Claudius tries to pray for forgiveness for his actions\u2014but admits that he knows that without truly repenting in deed as well as in word, he will never be saved or absolved. Claudius, for all his words of regret, doesn\u2019t actually feel regret about what he\u2019s done\u2014he\u2019s happy to be on the throne, to be married to Gertrude, and to have power over Denmark. However, Claudius\u2019s illegitimate rule is the \u201crotten\u201d core of the country, and as relationships within Elsinore splinter and fall apart, Claudius\u2019s bogus claim to the throne begins to threaten Denmark\u2019s political viability against its rivals in Norway. Self-serving, cool, and relentlessly ambitious, Claudius\u2019s arc ties in with several of the play\u2019s major themes, including action and inaction, appearance versus reality, and poison, corruption, and death.",
            "Dead King Hamlet's brother who has usurped the throne and married his sister-in-law."
        ]
    },
    {
        "character": "Gertrude",
        "descriptions": [
            "The Queen of Denmark, Hamlet\u2019s mother, recently married to Claudius. Gertrude loves Hamlet deeply, but she is a shallow, weak woman who seeks affection and status more urgently than moral rectitude or truth.",
            "Old Hamlet's widow and Claudius' wife.  She seems unaware that Claudius killed her former husband.  Gertrude loves Hamlet tremendously, while Hamlet has very mixed feelings about her for marrying the (in his eyes) inferior Claudius after her first husband's death.  Hamlet attributes this need for a husband to her lustiness.  Gertrude figures prominently in many of the major scenes in the play, including the killing of Polonius and the death of Ophelia.",
            "Hamlet\u2019s mother, Claudius\u2019s wife, and the Queen of Denmark. One of only two female characters in the play (along with Ophelia), Gertrude\u2019s arc throughout the drama is perhaps most representative of the theme of women. Gertrude marries her brother-in-law, Claudius, very shortly after the death of her husband\u2014an action perceived as cruel, obscene, lustful, and opportunistic by her son, Hamlet. In reality, this choice is one of the few options available to a woman of Gertrude\u2019s time: a woman whose political safety and social standing is entirely dependent on her connection to and protection from a powerful man. Gertrude, then, is one of the play\u2019s most complex characters, and one whose motives and truest nature are often obscured. For instance, when Hamlet confronts Gertrude about her choices, she admits that looking inward at her own choices\u2014and considering the idea that her new husband murdered her old one\u2014is too painful. Whether Gertrude is aware of Claudius\u2019s murderous ambition on any level remains up for debate, but what is clear is that no matter how much she knows, she is willing to tamp it down in order to selfishly secure her own sociopolitical safety. Gertrude\u2019s arc also ties in with the play\u2019s theme of action and inaction\u2014if, as the text overwhelmingly suggests, Gertrude knew that her husband was murdered by his brother and married Claudius anyway, then that choice is perhaps the play\u2019s most profound example of how both action and inaction can have complex moral implications.",
            "Prince Hamlet's mother, King Hamlet's widow, King Claudius' wife."
        ]
    }
]

# 출력 저장 리스트
all_outputs = []

# 각 캐릭터에 대해 프롬프트 요청
for entry in character_data:
    character = entry["character"]
    descriptions = entry["descriptions"]

    # 프롬프트 구성
    merged_descriptions = "\n".join(f"- {desc}" for desc in descriptions)
    full_prompt = (
        f"Describe character: {character} by merging the following descriptions into a single description.\n\n"
        f"Descriptions:\n{merged_descriptions}"
    )

    # GPT 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        n=1
    )

    content = response.choices[0].message.content.strip()
    print(f"\n--- {character} ---\n{content}\n")

    all_outputs.append({
        "character": character,
        "merged_description": content
    })

# 최종 결과 저장
output_path = "data/description/hamlet_description_merged.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for entry in all_outputs:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + "\n")