# TODO – UI improvements + Tab 5 (Monitoring & Feedback)

## Plan summary
- Add dedicated Tab 5: Monitoring & Feedback Loop.
- Standardize per-tab “story” layout for clearer educational guidance.
- Improve each existing tab’s structure: objective/inputs/outputs/interpretation/failures.

## Steps
1. Create new file `app/tabs/monitoring_feedback.py` implementing Tab 5 UI.
2. Update `main.py` to add Tab 5 mapping and sidebar entry. ✅
3. Extend `ml/storage.py` UI-read helpers only if needed (not expected for minimal implementation).
4. Smoke test:

   - `streamlit run main.py`
   - Ensure Tab 5 loads; feedback saving works without crashing.
   - Ensure existing 4 tabs still render.


