## 1. Relational Map (Edges)

Below, "→" indicates "leads to"/"is related to"; | is a label for the relationship.

**Major fields mapped**: consistency_tag, emotion_primary, immersion_level, descriptive_code

- historical_plausibility → supports → immersion_high
- period_appropriate_speech → supports → immersion_high
- dialogue_repetition → leads to → boredom/immersion_low
- flat_response → leads to → disappointment/frustration/immersion_low
- contradiction → leads to → confusion/disappointment/immersion_low
- character_coherence → supports → immersion_medium/high
- logical response → enables → interest/immersion_medium
- emotionless/AI-like tone → leads to → boredom/disappointment/immersion_low
- empathy/contextual accuracy → boosts → joy/immersion_high
- character inconsistency → leads to → disappointment/immersion_low
- dynamic interaction → boosts → interest/immersion_high
- in-character response → supports → interest/immersion_high
- relevant/natural dialogue → supports → joy/immersion_high
- awkward/irrelevant response → leads to → confusion/immersion_low
- stubbornness/excess consistency → leads to → frustration/immersion_low

---

## 2. Propositions

1. If **NPC dialogue is historically plausible or period-appropriate**, then **player immersion increases (immersion_high)**, mediated by **character_coherence and contextual fit**.
2. If there is **dialogue repetition or flat (emotionless) response**, then **player boredom or disappointment increases, and immersion decreases**, mediated by **lack of emotional dynamics and response variance**.
3. If **contradictions or character inconsistencies are present**, then **confusion and disappointment rise, and immersion falls**, mediated by **breaks in narrative logic and player trust**.
4. If **NPCs adapt dynamically and interact meaningfully**, then **player interest and joy increase, enhancing immersion**, mediated by **responsiveness and personalized engagement**.
5. If **NPC responses show empathy or contextual appropriateness**, then **players experience more positive emotions and greater immersion**, mediated by **emotional resonance and narrative plausibility**.

---

## 3. Korean Quotes Illustrating Distinct Links

**A. Historical/Character Consistency supports High Immersion**
- "캐릭터가 상황에 맞게 발끈하는 등 적절하게 대답했기 때문에." (`P01_1085_01` — period_appropriate_speech → joy/immersion_high)

**B. Dialogue Repetition leads to Boredom, Low Immersion**
- "계속해서 정의만을 이야기해, 같은 말을 반복하는 느낌이 들었다." (`P08_4421_01` — dialogue_repetition → boredom/immersion_low)

**C. Contradiction leads to Confusion, Low Immersion**
- "아까는 내 상황을 안다고 하더니 설명해보라니까 갑자기 모른다고 말을 바꾼다." (`P21_4614_01` — contradiction → confusion/immersion_low)