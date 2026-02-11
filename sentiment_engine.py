"""
sentiment_engine.py
───────────────────
Offline Sentiment & Emotion Detection
  • Polarity   : positive | negative | neutral   (with confidence 0-1)
  • Emotion    : joy | anger | sadness | fear | surprise | neutral
  • Subjectivity: objective | subjective

Uses a small hand-curated labelled corpus + TF-IDF + Logistic Regression
so the model trains in < 1 second on first import, no network needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ══════════════════════════════════════════════════════════════
# LABELLED TRAINING CORPUS  (expanded for reasonable accuracy)
# ══════════════════════════════════════════════════════════════
_SENTIMENT_DATA: list[tuple[str, str]] = [
    # ── positive ──────────────────────────────────────────
    ("I love this, it is amazing!", "positive"),
    ("Great job, wonderful work!", "positive"),
    ("This is fantastic and awesome", "positive"),
    ("I am so happy and excited today", "positive"),
    ("Brilliant, absolutely perfect", "positive"),
    ("Thank you so much, really appreciate it", "positive"),
    ("Best experience ever, highly recommend", "positive"),
    ("Outstanding performance, superb quality", "positive"),
    ("I feel great, everything is going well", "positive"),
    ("Wonderful, this made my day", "positive"),
    ("Excellent service, very impressed", "positive"),
    ("Beautiful and delightful, truly nice", "positive"),
    ("Loving every moment, so pleased", "positive"),
    ("Fantastic results, very satisfied", "positive"),
    ("Happy and grateful for everything", "positive"),
    ("This is top notch, really good", "positive"),
    ("Amazing features, love it so much", "positive"),
    ("Perfect solution, works great", "positive"),
    ("Delighted with the outcome, well done", "positive"),
    ("Incredible, beyond my expectations", "positive"),
    ("Joy and bliss fill my heart today", "positive"),
    ("Wonderful news, I am thrilled", "positive"),
    ("Superb and magnificent, truly excellent", "positive"),
    ("Feel so lucky and blessed right now", "positive"),
    ("Great vibes, everything is wonderful", "positive"),
    # extra positive – joy-heavy phrases
    ("Just got promoted, feeling elated and proud", "positive"),
    ("Celebrating, could not be happier about this", "positive"),
    ("The kids are laughing, makes me smile so much", "positive"),
    ("Finally finished, feeling accomplished and proud", "positive"),
    ("Achieved my goal at last, pure bliss", "positive"),
    ("Won the competition, feeling victorious and gleeful", "positive"),
    ("Grateful for this beautiful moment of peace", "positive"),
    ("Laughing so hard, feeling fantastic today", "positive"),
    ("Life is wonderful and full of exciting possibilities", "positive"),
    ("Woke up feeling refreshed and energetic this morning", "positive"),

    # ── negative ──────────────────────────────────────────
    ("I hate this, it is terrible", "negative"),
    ("Awful experience, very disappointing", "negative"),
    ("This is horrible and frustrating", "negative"),
    ("I am so angry and upset right now", "negative"),
    ("Worst product ever, total failure", "negative"),
    ("Terrible service, very unhappy", "negative"),
    ("Disappointed and annoyed, not good", "negative"),
    ("Bad quality, waste of money", "negative"),
    ("I feel sad and depressed today", "negative"),
    ("Horrible, never doing this again", "negative"),
    ("Disgusting experience, absolutely awful", "negative"),
    ("Very poor and unacceptable, pathetic", "negative"),
    ("Miserable and rotten, hate everything", "negative"),
    ("Dreadful outcome, extremely bad", "negative"),
    ("Furious and irate, unbelievable mistake", "negative"),
    ("Tragic and unfortunate, very bad news", "negative"),
    ("Painful and horrible, total disaster", "negative"),
    ("Annoyed and irritated by this mess", "negative"),
    ("Regret this decision, so unhappy", "negative"),
    ("Gloomy and bleak, nothing works", "negative"),
    ("Terrible and painful, cannot stand it", "negative"),
    ("Lousy and pathetic, completely failed", "negative"),
    ("Dumb decision, wasted all the time", "negative"),
    ("Horrible mistake, feeling so bad", "negative"),
    ("Everything went wrong, total nightmare", "negative"),
    # extra negative – fear / sadness / anger phrases
    ("I am so scared and anxious about what will happen", "negative"),
    ("Terrified and frightened, dread fills my heart", "negative"),
    ("Trembling with fear, something bad is coming", "negative"),
    ("Panicked and overwhelmed by the looming threat", "negative"),
    ("Anxious and worried, cannot calm down at all", "negative"),
    ("I feel so sad and lonely, tears are falling", "negative"),
    ("Broken hearted and sobbing uncontrollably today", "negative"),
    ("Grief and sorrow consume me, feeling devastated", "negative"),
    ("Depressed and hopeless, the pain will not stop", "negative"),
    ("Mourning the loss, heartache is unbearable now", "negative"),
    ("They broke their promise and I am furious", "negative"),
    ("Livid and outraged at the total incompetence", "negative"),
    ("Bitter resentment after being completely betrayed", "negative"),
    ("Seething with anger, this is completely unacceptable", "negative"),
    ("Nightmares and dread keep me awake every night", "negative"),

    # ── neutral ───────────────────────────────────────────
    ("The meeting is at three o clock", "neutral"),
    ("I need to buy groceries later", "neutral"),
    ("The report is due on Friday", "neutral"),
    ("Please send me the document", "neutral"),
    ("The weather forecast says cloudy", "neutral"),
    ("I will check back with you", "neutral"),
    ("The package arrives tomorrow morning", "neutral"),
    ("Can you schedule the appointment", "neutral"),
    ("The file has been updated now", "neutral"),
    ("Let me know when you are free", "neutral"),
    ("The project deadline is next week", "neutral"),
    ("I have noted your request", "neutral"),
    ("The system is running as expected", "neutral"),
    ("Please review the attached document", "neutral"),
    ("The data has been processed successfully", "neutral"),
    ("I will get back to you shortly", "neutral"),
    ("The order status is pending", "neutral"),
    ("Here is the information you asked for", "neutral"),
    ("The meeting has been rescheduled", "neutral"),
    ("Please confirm your availability", "neutral"),
    ("The task is completed as requested", "neutral"),
    ("I have forwarded the details", "neutral"),
    ("The update will be available soon", "neutral"),
    ("Please review and approve at your convenience", "neutral"),
    ("The inventory count is accurate", "neutral"),
    # extra neutral – factual / surprise-tone without emotion
    ("Wow I did not see that coming, completely shocked", "neutral"),
    ("Dropped my jaw, the announcement blindsided me", "neutral"),
    ("That is an unexpected and surprising development", "neutral"),
    ("The plot twist caught everyone off guard today", "neutral"),
    ("Just got the best news ever, feeling on top", "neutral"),
    ("The unknown territory lies ahead of us now", "neutral"),
    ("Startling discovery was made by the research team", "neutral"),
    ("The sudden change of plans affects everyone here", "neutral"),
    ("An unforeseen event occurred during the event", "neutral"),
    ("The results were not what anyone had predicted", "neutral"),
]

_EMOTION_DATA: list[tuple[str, str]] = [
    # ── joy ───────────────────────────────────────────────
    ("I am so happy and joyful today!", "joy"),
    ("This is wonderful, I feel amazing", "joy"),
    ("Ecstatic and thrilled with everything", "joy"),
    ("Great news, I am so excited", "joy"),
    ("Love it, feeling blessed and happy", "joy"),
    ("Delighted and cheerful, what a day", "joy"),
    ("Fantastic, pure happiness right now", "joy"),
    ("So grateful and full of joy today", "joy"),
    ("Wonderful surprise, I am overjoyed", "joy"),
    ("Best day ever, feeling on top of the world", "joy"),
    ("Just got promoted, feeling elated and proud", "joy"),
    ("Celebrating our anniversary, could not be happier", "joy"),
    ("The kids are laughing and playing, makes me smile", "joy"),
    ("Finally finished the project, feeling accomplished and proud", "joy"),
    ("Sunny beautiful day, enjoying every minute of it", "joy"),
    ("Got great results back, so pleased and relieved", "joy"),
    ("My favourite song came on, instantly cheered up", "joy"),
    ("Reunited with old friends, feeling warm and content", "joy"),
    ("The surprise party was incredible, feeling loved", "joy"),
    ("Achieved my goal at last, pure bliss and satisfaction", "joy"),
    ("Everything is going perfectly, life is good", "joy"),
    ("Woke up feeling refreshed and energetic this morning", "joy"),
    ("Such a pleasant and enjoyable evening overall", "joy"),
    ("Puppy videos always make me feel cheerful and bright", "joy"),
    ("Thankful and appreciative of all the good things", "joy"),
    ("Won the competition, feeling victorious and gleeful", "joy"),
    ("Had the most fun time with my family today", "joy"),
    ("Grateful for this beautiful moment of peace and calm", "joy"),
    ("Laughing so hard my stomach hurts, feeling fantastic", "joy"),
    ("Life is wonderful and full of exciting possibilities", "joy"),

    # ── anger ─────────────────────────────────────────────
    ("I am furious and so angry right now", "anger"),
    ("This is infuriating, absolutely outrageous behaviour", "anger"),
    ("Completely unacceptable, makes me livid with rage", "anger"),
    ("I cannot believe this happened, so annoyed", "anger"),
    ("Extremely frustrated and irate about the whole situation", "anger"),
    ("Angry and disgusted, this is totally wrong", "anger"),
    ("Terrible treatment from them, I am outraged", "anger"),
    ("Furious beyond belief, an unforgivable mistake was made", "anger"),
    ("So irritated and completely fed up with this nonsense", "anger"),
    ("Rage and fury are boiling inside me right now", "anger"),
    ("They keep ignoring me and it makes me seething mad", "anger"),
    ("The service was appalling, I demand an explanation", "anger"),
    ("Livid about the delay, this is unacceptable conduct", "anger"),
    ("Absolutely fed up and furious with the management", "anger"),
    ("Cannot stand this incompetence, it drives me crazy", "anger"),
    ("The rudeness was shocking, I am beyond irritated", "anger"),
    ("Disgusted by the lack of effort, very disappointed", "anger"),
    ("They broke their promise and now I am furious", "anger"),
    ("This whole mess is making me lose my temper", "anger"),
    ("Enraged and bitter about how this was all handled", "anger"),
    ("Angry beyond words at their careless attitude", "anger"),
    ("The betrayal makes me boil with indignation", "anger"),
    ("Maddened by the repeated failures and excuses", "anger"),
    ("Wrath and contempt are all I feel right now", "anger"),
    ("Incensed by the injustice of the entire situation", "anger"),
    ("Hostility and resentment building up inside me", "anger"),
    ("Irate and exasperated, nothing is being done about this", "anger"),
    ("Outraged that they got away with such behaviour", "anger"),
    ("Simmering with anger after that disastrous meeting", "anger"),
    ("Bitter and resentful, they have ruined everything", "anger"),

    # ── sadness ───────────────────────────────────────────
    ("I feel so sad and lonely today", "sadness"),
    ("This is heartbreaking and deeply depressing news", "sadness"),
    ("Tears in my eyes, feeling utterly miserable", "sadness"),
    ("So unhappy and feeling really down right now", "sadness"),
    ("Deeply saddened by the tragic news today", "sadness"),
    ("Feeling blue and completely hopeless about everything", "sadness"),
    ("Grief and sorrow fill my heavy heart", "sadness"),
    ("Lost and sad, nothing seems right anymore", "sadness"),
    ("Melancholy and gloomy, the pain really hurts", "sadness"),
    ("Crying and feeling totally devastated today", "sadness"),
    ("Missing someone who is no longer here, feeling empty", "sadness"),
    ("The loss is unbearable, drowning in sorrow", "sadness"),
    ("Everything reminds me of better times, feeling wistful", "sadness"),
    ("Heartache and disappointment linger deep inside me", "sadness"),
    ("Alone in the dark, consumed by deep sadness", "sadness"),
    ("The rejection stings and leaves me feeling crushed", "sadness"),
    ("Mourning the end of something truly precious", "sadness"),
    ("A heavy weight sits on my chest, feeling hopeless", "sadness"),
    ("Tears stream down my face, feeling broken inside", "sadness"),
    ("The world feels grey and colourless today", "sadness"),
    ("Regret and remorse make my heart ache deeply", "sadness"),
    ("Painful memories flood back and bring me to tears", "sadness"),
    ("Feeling defeated and utterly drained of all hope", "sadness"),
    ("The silence around me amplifies my loneliness", "sadness"),
    ("A deep ache of sadness that will not go away", "sadness"),
    ("Broken hearted and sobbing uncontrollably right now", "sadness"),
    ("Depressed and withdrawn, cannot face the world today", "sadness"),
    ("Sorrowful and mournful, missing the good old days", "sadness"),
    ("Tears of pain and deep emotion will not stop", "sadness"),
    ("Feeling desolate and completely without any hope", "sadness"),

    # ── fear ──────────────────────────────────────────────
    ("I am terrified and scared of this situation", "fear"),
    ("So anxious and worried about what happens tomorrow", "fear"),
    ("This frightens me deeply, feeling very nervous", "fear"),
    ("Dread and panic, I just cannot calm down", "fear"),
    ("Worried and fearful about the uncertain outcome ahead", "fear"),
    ("It feels threatening and really scary to me", "fear"),
    ("Anxious and uneasy, the fear keeps creeping in", "fear"),
    ("Horrified and shocked, feeling completely unsafe", "fear"),
    ("Terror and apprehension, I am very afraid", "fear"),
    ("Nervous and on edge, deeply worried all day", "fear"),
    ("The uncertainty is killing me, paralysed by fear", "fear"),
    ("Heart racing with panic at the thought of it", "fear"),
    ("Trembling with dread, something bad might happen", "fear"),
    ("Nightmares haunt me, living in constant fear", "fear"),
    ("Phobia grips me tight, cannot stop the anxiety", "fear"),
    ("Dreadful foreboding, something feels terribly wrong", "fear"),
    ("Stomach in knots, overwhelmed by nervousness", "fear"),
    ("The looming threat makes me feel powerless and scared", "fear"),
    ("Claustrophobic and panicked, walls closing in on me", "fear"),
    ("Frightened to the bone, cannot shake the dread", "fear"),
    ("Apprehensive and uneasy, expecting the worst to come", "fear"),
    ("Spine tingling with fear at every loud noise", "fear"),
    ("Overwhelmed by worry and irrational but consuming fear", "fear"),
    ("The unknown terrifies me more than anything else", "fear"),
    ("Shaking with anxiety, unable to think clearly", "fear"),
    ("Petrified of what lies ahead in the darkness", "fear"),
    ("Cold sweat and racing heart, gripped by pure terror", "fear"),
    ("Trepidation fills every fibre of my exhausted being", "fear"),
    ("Cannot sleep because the fear will not subside", "fear"),
    ("Vulnerable and exposed, drowning in helpless fear", "fear"),

    # ── surprise ──────────────────────────────────────────
    ("Wow, I absolutely did not see that coming at all", "surprise"),
    ("That is shocking news, completely and utterly unexpected", "surprise"),
    ("Amazed and stunned, I simply cannot believe it", "surprise"),
    ("Totally surprised by this wild revelation today", "surprise"),
    ("Astonished and completely blown away right now", "surprise"),
    ("Never in a million years expected this, what a shock", "surprise"),
    ("Speechless and stunned, this is incredible shocking news", "surprise"),
    ("Startled and taken completely aback by the twist", "surprise"),
    ("Mind blown, I absolutely did not expect any of this", "surprise"),
    ("Unbelievable turn of events, totally shocked and stunned", "surprise"),
    ("Dropped my jaw, the announcement caught me off guard", "surprise"),
    ("Cannot wrap my head around it, so unexpected", "surprise"),
    ("The plot twist was the most shocking thing ever", "surprise"),
    ("Baffled and bewildered, this defies all expectations", "surprise"),
    ("Out of nowhere it happened, leaving me speechless", "surprise"),
    ("Stunned into silence by the sudden revelation", "surprise"),
    ("Eyes wide open, shocked to my very core", "surprise"),
    ("The twist blindsided everyone in the entire room", "surprise"),
    ("Completely caught off guard, gasping in disbelief", "surprise"),
    ("Jaw dropping moment, never saw it coming at all", "surprise"),
    ("Flabbergasted by the turn the story has taken", "surprise"),
    ("Reeling from shock, the news hit like a thunderbolt", "surprise"),
    ("Thunderstruck and dumbfounded by the revelation", "surprise"),
    ("Incredulous and wide eyed at the unexpected outcome", "surprise"),
    ("The bombshell news left me utterly speechless today", "surprise"),
    ("Startling discovery that no one could have predicted", "surprise"),
    ("Gobsmacked by how things turned out in the end", "surprise"),
    ("Gasping at the sheer unexpectedness of the event", "surprise"),
    ("Everything changed in an instant, totally blindsided", "surprise"),
    ("Bewildered and astonished, reality feels surreal now", "surprise"),

    # ── neutral ───────────────────────────────────────────
    ("I will attend the meeting later today", "neutral"),
    ("The document has been received and logged", "neutral"),
    ("Please confirm the details when you can", "neutral"),
    ("The task is currently in progress", "neutral"),
    ("I noted your message and will act on it", "neutral"),
    ("Will update you shortly on the status", "neutral"),
    ("The data is being processed right now", "neutral"),
    ("Sending the report to you now", "neutral"),
    ("The schedule has been set for next week", "neutral"),
    ("Acknowledged, will follow up as planned", "neutral"),
    ("The package should arrive by Thursday", "neutral"),
    ("Please review the attached file at your convenience", "neutral"),
    ("The system is operating within normal parameters", "neutral"),
    ("I have forwarded the information to the team", "neutral"),
    ("The meeting has been moved to next Tuesday", "neutral"),
    ("Here is the summary you requested earlier", "neutral"),
    ("The order is being processed as we speak", "neutral"),
    ("I will check on that and get back to you", "neutral"),
    ("The inventory has been updated accordingly", "neutral"),
    ("Please submit the form before the deadline", "neutral"),
    ("The report covers data from last quarter", "neutral"),
    ("I have noted your preference for the record", "neutral"),
    ("The server maintenance is scheduled for tonight", "neutral"),
    ("Your account details have been updated successfully", "neutral"),
    ("The shipment is currently in transit", "neutral"),
    ("I will arrange for that to be done", "neutral"),
    ("The configuration has been saved as requested", "neutral"),
    ("Please let me know if you need anything else", "neutral"),
    ("The project timeline remains on track so far", "neutral"),
    ("The file has been uploaded to the shared drive", "neutral"),
]


# ══════════════════════════════════════════════════════════════
# PIPELINES  (fit once at module load – ~0.3 s)
# ══════════════════════════════════════════════════════════════
def _build_pipeline(data: list[tuple[str, str]]) -> Pipeline:
    texts, labels = zip(*data)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
    ])
    pipe.fit(list(texts), list(labels))
    return pipe

_SENTIMENT_PIPE: Pipeline = _build_pipeline(_SENTIMENT_DATA)
_EMOTION_PIPE  : Pipeline = _build_pipeline(_EMOTION_DATA)


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════
class SentimentAnalyser:
    """Single entry-point for all sentiment / emotion features."""

    @staticmethod
    def analyse(text: str) -> dict:
        """
        Returns:
            {
                "polarity":      "positive" | "negative" | "neutral",
                "polarity_conf": float  (0-1),
                "emotion":       "joy" | "anger" | "sadness" | "fear" | "surprise" | "neutral",
                "emotion_conf":  float  (0-1),
                "subjectivity":  "subjective" | "objective",
                "emoji":         str   (representative emoji)
            }
        """
        if not text or not text.strip():
            return _default_result()

        # ── polarity ──────────────────────────────────
        pol_probs  = _SENTIMENT_PIPE.predict_proba([text])[0]
        pol_classes = list(_SENTIMENT_PIPE.classes_)
        pol_idx    = int(np.argmax(pol_probs))
        polarity   = pol_classes[pol_idx]
        pol_conf   = round(float(pol_probs[pol_idx]), 3)

        # ── emotion ───────────────────────────────────
        emo_probs  = _EMOTION_PIPE.predict_proba([text])[0]
        emo_classes = list(_EMOTION_PIPE.classes_)
        emo_idx    = int(np.argmax(emo_probs))
        emotion    = emo_classes[emo_idx]
        emo_conf   = round(float(emo_probs[emo_idx]), 3)

        # ── polarity ←→ emotion consistency fix ──────────
        # If emotion is clearly negative (anger/fear/sadness) but polarity
        # landed on neutral with low confidence, trust the emotion signal.
        _NEG_EMOTIONS = {"anger", "fear", "sadness"}
        if emotion in _NEG_EMOTIONS and polarity == "neutral" and pol_conf < 0.70:
            polarity  = "negative"
            pol_conf  = round(max(pol_conf, emo_conf * 0.85), 3)

        # ── subjectivity heuristic ────────────────────
        subj_words = {"feel","love","hate","amazing","terrible","happy","sad",
                      "great","awful","wonderful","horrible","excited","angry",
                      "beautiful","ugly","best","worst","enjoy","dislike"}
        tokens     = set(text.lower().split())
        subjectivity = "subjective" if tokens & subj_words else "objective"

        return {
            "polarity":      polarity,
            "polarity_conf": pol_conf,
            "emotion":       emotion,
            "emotion_conf":  emo_conf,
            "subjectivity":  subjectivity,
            "emoji":         _EMOJI_MAP.get(emotion, "😐"),
        }

    # ── batch helper ──────────────────────────────────────
    @staticmethod
    def analyse_batch(texts: list[str]) -> list[dict]:
        return [SentimentAnalyser.analyse(t) for t in texts]


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
_EMOJI_MAP = {
    "joy":      "😄",
    "anger":    "😠",
    "sadness":  "😢",
    "fear":     "😨",
    "surprise": "😲",
    "neutral":  "😐",
}

def _default_result() -> dict:
    return {
        "polarity": "neutral", "polarity_conf": 0.5,
        "emotion": "neutral",  "emotion_conf": 0.5,
        "subjectivity": "objective", "emoji": "😐",
    }