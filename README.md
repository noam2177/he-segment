# he-segment

פיצול תחיליות עבריות מהגזע. שלוש מערכות על אותו זהב: בייסליין, לקסיקון שנבנה רק מ-train, ומודל תווים שאומן רק על train.

על שורות ה-test שבצילום הזה: בייסליין 21 מתוך 41, לקסיקון אותו מספר (המילים המוגנות נשארו ב-train), מודל התווים 32 מתוך 41. הפירוט ב-`fixtures/report.json`. הסט עוד לא נעול.

```text
python -m unittest discover -s tests -q
python -m he_segment.report
```

השלבים, מהסוף אחורה, ב-`PLAN.md`.
