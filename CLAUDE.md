# Engineering RAG Assistant — Project Context

> מסמך הקשר קבוע ל-Claude Code. המפרט המלא ב-`PROJECT_1_SPEC.md`.

## מה זה
עוזר RAG שעונה על שאלות הנדסיות מתוך בסיס ידע של מסמכים (תקנים, דאטה-שיטים, מדריכי תכן). גרסת קוד מלאה של Custom GPT שהמשתמש בנה כמהנדס מערכות ברפאל. **המטרה האמיתית:** פרויקט portfolio שמדגים RAG, embeddings, prompt engineering, ו-Claude API integration למגייס.

## Stack
- **Python 3.11+**
- **Anthropic SDK** — Claude API לגנרציה
- **sentence-transformers** מקומי (`BAAI/bge-small-en-v1.5`, 384-dim) — embeddings. בחרנו מקומי על פני Voyage AI: אפס תלות חיצונית, אפס עלות per-query, הרצה אופליין מלאה (חשוב למסמכים הנדסיים סודיים). Trade-off: איכות מעט נמוכה ממודל hosted, אנגלית בלבד
- **Qdrant** ב-Docker — vector DB (נבחר על פני Chroma/Pinecone: production-grade, מקומי, אפס עלות)
- **FastAPI** — backend עם `/docs` אוטומטי
- **Streamlit** — UI לפרויקט הזה (React בפרויקט הבא)
- **docker-compose** — Qdrant + backend ביחד

## ארכיטקטורה
שני pipelines נפרדים בכוונה:
- **Ingestion (offline):** PDFs → chunking → embeddings → Qdrant
- **Query (online):** שאלה → embed → top-k search → prompt עם context → Claude → תשובה + מקורות

## מבנה הריפו
```
src/
├── ingest.py         pipeline להעלאת מסמכים
├── chunking.py       פיצול לקטעים (~500 tokens, overlap 50)
├── embeddings.py     wrapper סביב מודל embedding
├── vectorstore.py    כל האינטראקציה עם Qdrant
├── retrieval.py      semantic search
├── prompts.py        תבניות prompt (מבודד בכוונה)
├── generate.py       קריאה ל-Claude עם context
└── api.py            FastAPI endpoints
app.py                Streamlit UI
eval/
├── test_questions.json  15-20 שאלות עם תשובות צפויות
└── run_eval.py          מודד accuracy/relevance
tests/
└── test_retrieval.py
```

## סדר משימות אג'נטי
כל משימה = יחידה שלמה. אחריה **המשתמש בודק** לפני שממשיכים (זה ה-skill של "validate AI-generated work").

0. **Scaffolding** — מבנה + docker-compose + requirements + .env.example. רק שלד עם TODOs.
1. **Ingestion + Chunking** — קריאת PDFs, פיצול, metadata (שם קובץ + עמוד).
2. **Embeddings + Vector Store** — embed + אחסון ב-Qdrant + פונקציית search.
3. **Generation + Prompts** — prompt שמזריק context, מורה לענות רק מהמקורות, ולומר "אין מידע" אם אין. **המקרה הקריטי:** למנוע הזיות.
4. **API + UI** — FastAPI `/query` + Streamlit (שאלה, תשובה, מקורות מתקפלים).
5. **Evaluation** — `run_eval.py` מודד accuracy על test_questions.json.

## עקרונות עבודה
- **אל תכתוב קוד פרויקט עד שהמשתמש מאשר משימה ספציפית.** המסמך הזה לא הזמנה להתחיל לבנות.
- **אחרי כל משימה — עצור.** המשתמש צריך לבדוק לפני שממשיכים. זה לא bottleneck, זה חלק מהפרויקט.
- **הפרדה נקייה בין קבצים.** Ingestion/retrieval/prompts/generation בקבצים נפרדים — מגייס שרואה את זה מבין שאנחנו חושבים על ארכיטקטורה.
- **תעד החלטות הנדסיות.** למה chunk size כזה? למה Qdrant? אלו השאלות בראיון — התשובות חיות ב-README ובהערות קצרות בקוד.
- **טפל ב"לא יודע" כדרישה מהותית, לא edge case.** זה ה-"AI limitations" שהתפקיד מבקש.

## החלטות שצריך להגן עליהן (לראיון)
1. chunk size 500 tokens, overlap 50 — איזון בין הקשר ודיוק retrieval
2. אנטי-הזיות: prompt מחמיר + הוראה לומר "אין מידע" + ציטוט מקורות
3. Qdrant על פני חלופות — production-grade, containerized, שליטה מלאה
4. **Embeddings מקומיים** (`bge-small-en-v1.5`) על פני Voyage/OpenAI — אוטונומיה, אפס עלות, אפס תלות רשת, מתאים למסמכים סודיים. עלות: דיוק תיאורטי מעט נמוך יותר ואנגלית בלבד
5. מדידה אמיתית עם eval set — לא "בניתי RAG" אלא "85% accuracy על 20 שאלות"
6. מגבלות מוכרות — אין re-ranking, אין conversational memory, retrieval תלוי ב-chunking

## שדרוגים אופציונליים (רק אחרי שהבסיס עובד)
Re-ranking · Hybrid search · Streaming responses · Conversational memory

## ה-README הוא המוצר
המגייס יקרא README לפני שיריץ שורת קוד. חובה: משפט פתיחה שמחבר לסיפור של המשתמש, GIF דמו, דיאגרמת ארכיטקטורה, הוראות הרצה, סעיף "Engineering Decisions", סעיף "Built with Claude Code", תוצאות eval.

## שפה
המשתמש מדבר עברית. ענה בעברית בשיחה. קוד, שמות משתנים, והערות בקוד — באנגלית.
