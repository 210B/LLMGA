## 1. Relational Map (Edges)

**Edge format:** source → relation → target

- period_appropriate_speech → supports → high_immersion
- period_appropriate_speech → elicits → joy/interest
- period_appropriate_speech → reveals → expected_character_traits
- historical_plausibility → supports → high_immersion
- historical_plausibility → elicits → joy/interest
- historical_plausibility → reveals → expected_character_traits
- flat_response → reduces → immersion
- flat_response → elicits → disappointment/boredom
- contradiction → reduces → immersion
- contradiction → elicits → disappointment/confusion/frustration
- contradiction → reveals → illogical_behavior
- dialogue_repetition → reduces → immersion/engagement
- dialogue_repetition → elicits → boredom/disappointment
- dialogue_repetition → reveals → generic_NPC_behavior
- none (inconsistency_tag) → links to → mixed/neutral_emotion OR curiosity
- natural_tone/interaction (implied in some "none") → supports → immersion/curiosity

---

## 2. Propositions

1. **If** NPC speech is period-appropriate or historically plausible, **then** user immersion is high and positive emotions (joy, interest) increase, **mediated by** alignment with character expectations and story context.
2. **If** NPC responses are flat, overly abstract, or unresponsive, **then** user immersion and engagement decrease and negative emotions (boredom, disappointment) arise, **mediated by** perceived lack of interactivity or depth.
3. **If** NPC dialogue is contradictory or lacks logical coherence, **then** immersion is broken and confusion or frustration occurs, **mediated by** expectations of character and narrative consistency.
4. **If** NPCs use repetitive dialogue or maintain excessive behavioral consistency, **then** user engagement and immersion decrease, **mediated by** perceived lack of character individuality and dynamic interaction.
5. **If** dialogue and character actions remain consistent with known personalities and narrative background, **then** immersion and believability are enhanced, **mediated by** fulfillment of user expectations and narrative plausibility.

---

## 3. Korean Quotes (Distinct Links)

1. **Period-Appropriate Speech → High Immersion**
   > "해당 캐릭터에 맞는 대사를 했다고 생각했다."  
   (*P03, period_appropriate_speech → supports → immersion*)

2. **Flat Response → Reduced Engagement**
   > "소통한다는 느낌이 들지 않았다."  
   (*P21, flat_response → reduces → immersion*)

3. **Contradiction → Broken Immersion**
   > "자신이 범인이 아니라고 했었는데, 그 다음에 바로 자신이 범인이라고 말을 돌리는 부분에서 어색함을 느꼈습니다."  
   (*P04, contradiction → reduces → immersion*)

4. **Dialogue Repetition → Boredom**
   > "다 똑같은 말투를 사용한거 같습니다"  
   (*P23, dialogue_repetition → elicits → boredom*)

5. **Historical Plausibility → Joy/Believability**
   > "알고 있는 햄릿 내용에서 이질감이 크게 들지 않았다."  
   (*P14, historical_plausibility → supports → immersion and joy*)