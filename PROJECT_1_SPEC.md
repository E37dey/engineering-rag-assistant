# Project 1 — Engineering RAG Assistant

> מפרט עבודה ל-Claude Code. פתח את הקובץ הזה ב-VS Code לצד הריפו, והשתמש בו כ-`CLAUDE.md` או כמסמך ייחוס למשימות אג'נטיות.

---

## 1. מה אנחנו בונים ולמה

עוזר AI שעונה על שאלות הנדסיות מתוך בסיס ידע של מסמכים הנדסיים (תקנים, דאטה-שיטים, מדריכי תכן). זו גרסת הקוד האמיתית של ה-Custom GPT שכבר בנית ברפאל — אותו רעיון, אבל הפעם אתה הבעלים של כל שכבת התשתית.

**מה זה מוכיח למגייס (מיפוי ישיר לתיאור התפקיד):**

| דרישה בתפקיד | איפה זה בא לידי ביטוי |
|---|---|
| RAG, embeddings, vector databases | הליבה של המערכת |
| Prompt engineering | שכבת ה-prompt construction עם context injection |
| Claude API / LLM integration | קריאות ל-Anthropic SDK |
| Python, APIs | backend ב-FastAPI |
| Model evaluation, AI limitations | סט eval בסיסי + טיפול ב"לא יודע" |
| AI-assisted development | אתה בונה הכל עם Claude Code ומתעד את זה |

---

## 2. ה-Stack המדויק (וההצדקה לראיון)

- **Python 3.11+** — שפת הליבה
- **Anthropic SDK** (`anthropic`) — קריאות ל-Claude לתשובות
- **Embeddings**: `voyage-3` דרך Voyage AI (השותף המומלץ של Anthropic ל-embeddings), או חלופית `sentence-transformers` מקומי אם רוצים בלי תלות חיצונית
- **Vector DB**: **Qdrant** ב-Docker מקומי. *הצדקה לראיון:* production-grade, לא צעצוע למידה כמו Chroma; הרצה ב-Docker מראה הבנת containerization; מקומי = שליטה מלאה ב-deployment ואפס עלות ענן
- **Backend**: **FastAPI** — מהיר, async, עם תיעוד אוטומטי ב-`/docs` (מרשים בדמו)
- **Frontend**: **Streamlit** לגרסה הראשונה (מהיר לבנות, נראה מקצועי). בפרויקט 2 נעבור ל-React
- **Containerization**: `docker-compose` שמרים את Qdrant + ה-backend יחד

---

## 3. ארכיטקטורה

```
מסמכים (PDF) 
   │
   ▼
[Ingestion Pipeline]  ── chunking ──► embedding ──► אחסון ב-Qdrant
   
שאלת משתמש
   │
   ▼
[Retrieval]  ── embed query ──► חיפוש top-k ב-Qdrant ──► chunks רלוונטיים
   │
   ▼
[Prompt Construction]  ── הזרקת context + הוראות ──► Claude API
   │
   ▼
תשובה + ציטוט מקורות (מאיזה מסמך/עמוד)
```

**ההפרדה הזו חשובה.** Ingestion רץ פעם אחת (offline). Retrieval+generation רץ בכל שאלה (online). שמור אותם בקבצים נפרדים — מגייס שרואה הפרדה נקייה מבין שאתה חושב על ארכיטקטורה.

---

## 4. מבנה הריפו

```
engineering-rag/
├── README.md                  ← הכי חשוב למגייס (ראה סעיף 8)
├── docker-compose.yml         ← Qdrant + backend
├── requirements.txt
├── .env.example               ← מפתחות API (לעולם לא .env אמיתי ב-git)
├── src/
│   ├── ingest.py              ← pipeline להעלאת מסמכים
│   ├── chunking.py            ← פיצול מסמכים לקטעים
│   ├── embeddings.py          ← wrapper סביב מודל ה-embedding
│   ├── vectorstore.py         ← כל האינטראקציה עם Qdrant
│   ├── retrieval.py           ← חיפוש semantic
│   ├── prompts.py             ← תבניות prompt (מבודד בכוונה)
│   ├── generate.py            ← קריאה ל-Claude עם context
│   └── api.py                 ← FastAPI endpoints
├── app.py                     ← Streamlit UI
├── eval/
│   ├── test_questions.json    ← 15-20 שאלות עם תשובות צפויות
│   └── run_eval.py            ← מודד accuracy/relevance
└── tests/
    └── test_retrieval.py      ← unit tests בסיסיים
```

---

## 5. סדר עבודה אג'נטי (משימות ל-Claude Code)

הרעיון: כל משימה היא יחידה שסוכן יכול לבצע מקצה לקצה, ואז **אתה נכנס לבדוק** לפני שממשיכים. הבדיקה שלך = ה-skill ש"validate AI-generated work" שהתפקיד מבקש.

### משימה 0 — Scaffolding
> "צור את מבנה הריפו לפי הספק, כולל docker-compose עם Qdrant, requirements.txt, ו-.env.example. אל תכתוב לוגיקה עדיין — רק שלד עם TODOs."

**נקודת ביקורת שלך:** ודא ש-`docker-compose up` מרים את Qdrant ושאתה רואה את ה-dashboard שלו ב-`localhost:6333/dashboard`.

### משימה 1 — Ingestion + Chunking
> "ממש את ingest.py ו-chunking.py. קרא PDFs מתיקיית data/, פצל לקטעים של ~500 טוקנים עם overlap של 50, צרף metadata (שם קובץ, מספר עמוד)."

**נקודת ביקורת שלך:** הרץ על 2-3 PDFs ובדוק שהקטעים הגיוניים — לא חתוכים באמצע משפט קריטי. זו החלטה הנדסית אמיתית שתסביר בראיון: *למה 500 טוקנים? איך בחרת overlap?*

### משימה 2 — Embeddings + Vector Store
> "ממש את embeddings.py ו-vectorstore.py. embed את הקטעים ואחסן ב-Qdrant עם ה-metadata. הוסף פונקציית search שמחזירה top-k."

**נקודת ביקורת שלך:** חפש שאילתה ידנית וראה אם הקטעים שחוזרים באמת רלוונטיים. אם לא — זו בעיה אמיתית לפתור (chunk size? מודל embedding?).

### משימה 3 — Generation + Prompts
> "ממש את prompts.py ו-generate.py. בנה prompt שמזריק את ה-chunks כ-context, מורה ל-Claude לענות רק מהמקורות, ולומר 'אין לי מידע' אם אין. החזר גם את המקורות."

**נקודת ביקורת שלך — הכי חשובה:** בדוק את מקרה ה-"לא יודע". שאל שאלה שאין עליה תשובה בדאטה. אם המודל ממציא — תקן את ה-prompt. *זה בדיוק "ensure responsible use of AI" ו-"AI system limitations" מהתפקיד.*

### משימה 4 — API + UI
> "עטוף את הכל ב-FastAPI (endpoint /query) ובנה Streamlit UI עם שדה שאלה, תשובה, ורשימת מקורות מתקפלת."

**נקודת ביקורת שלך:** דמו מקצה לקצה. צלם וידאו קצר — זה הולך ל-README.

### משימה 5 — Evaluation
> "צור eval/run_eval.py שרץ על test_questions.json, ובודק לכל שאלה אם התשובה מכילה את המידע הצפוי ואם המקורות נכונים. הדפס accuracy."

**נקודת ביקורת שלך:** הרץ, קבל מספר. אם accuracy נמוך — שפר. *להגיד בראיון "ה-RAG שלי מגיע ל-85% accuracy על 20 שאלות בדיקה" שווה פי 100 מ"בניתי RAG".*

---

## 6. החלטות הנדסיות שתצטרך להגן עליהן בראיון

הכן תשובה קצרה לכל אחת — אלו השאלות שמגייס טכני ישאל:

1. **למה chunk size כזה?** — איזון בין הקשר מספיק (לא קטן מדי) ודיוק retrieval (לא גדול מדי שמכניס רעש).
2. **איך התמודדת עם הזיות?** — prompt מחמיר + הוראה מפורשת לומר "אין מידע" + החזרת מקורות לאימות.
3. **למה Qdrant ולא Chroma/Pinecone?** — ראה הצדקה בסעיף 2.
4. **איך מדדת שזה עובד?** — סט ה-eval שלך.
5. **מה המגבלות?** — אין re-ranking, אין conversational memory, retrieval תלוי באיכות ה-chunking. *להכיר במגבלות זה בדיוק "AI system limitations" מהתפקיד — זה מראה בגרות.*

---

## 7. שדרוגים אופציונליים (אם נשאר זמן ורוצים להבריק)

- **Re-ranking** של תוצאות ה-retrieval (cross-encoder) — שדרוג איכות מרשים
- **Hybrid search** (semantic + keyword) ב-Qdrant
- **Streaming** של התשובה ב-UI (כמו ChatGPT)
- **Conversational memory** — שאלות המשך

אל תתקע על אלו לפני שהבסיס עובד. ניצחון גמור > פיצ'רים חצי-עבודה.

---

## 8. ה-README — הנכס החשוב ביותר

המגייס יקרא את ה-README לפני שיריץ שורת קוד אחת. הוא חייב לכלול:

1. **משפט פתיחה** שמחבר לסיפור שלך: "עוזר RAG למסמכים הנדסיים — נבנה כגרסת קוד מלאה של פתרון Custom GPT שהטמעתי בעבודתי כמהנדס מערכות."
2. **GIF/וידאו קצר** של הדמו עובד
3. **דיאגרמת ארכיטקטורה** (אפשר אותה מסעיף 3)
4. **הוראות הרצה** (`docker-compose up` → `streamlit run app.py`)
5. **סעיף "Engineering Decisions"** — 4-5 ההחלטות מסעיף 6, כתובות בקצרה. *זה מה שמבדיל אותך ממישהו שהעתיק טוטוריאל.*
6. **סעיף "Built with Claude Code"** — תאר בקצרה איך השתמשת בסוכן, ואיך validate-ת את הקוד. *זה ממלא ישירות דרישה בתפקיד.*
7. **תוצאות ה-eval** — "85% accuracy on 20 test questions"

---

## 9. הערכת זמן

עם Claude Code וניסיון VS Code, בקצב של ~10-12 שעות:
- משימות 0-2: שבוע ראשון (התשתית)
- משימות 3-4: שבוע שני (הליבה + UI)
- משימה 5 + README: סוף שבוע שני / תחילת שלישי

**סה"כ: ~2-3 שבועות לפרויקט מלוטש עם README.**
