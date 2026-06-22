# 🎤 Presenter Demo Script (~8 minutes)

> Goal: walk the audience through the **entire ML lifecycle** using the attendance
> demo. Keep it visual. Read each yellow 💡 lesson aloud — they're the takeaways.

**Before you start:** run `streamlit run main.py` and pre-download the InsightFace
models. **Pro Tip:** Click `🚀 Load Sample Dataset` in the sidebar to instantly load 
historical scientists (Ada Lovelace, Alan Turing, Grace Hopper) with a pre-trained SVM 
model so you can showcase the downstream tabs instantly.

---

### 0 · Hook (30s)
> "In the next 5–8 minutes you'll see the complete lifecycle of a machine-learning 
> project — from processing raw data to building and deploying a classifier. 
> Our example: marking attendance by recognizing faces."

Point at the **sidebar**: 4 tabs = the complete streamlined pipeline.

---

### 1 · Data Processing (3 mins)  → *"ML learns from clean features, not raw photos"*
- Click on **1 · Data Processing** and show the three sub-tabs:
  - **Sub-Tab 1: Data Collection:** Type a name, capture a pose, click **Register this pose** (capture 2–3 poses).
  - **Sub-Tab 2: Data Preparation:** Show the **Brightness / Contrast / Blur / Flip** gallery multiplying samples.
  - **Sub-Tab 3: Feature Extraction:** Capture a face, view the 512-D identity vector strip, and scroll to see how embeddings cluster in the PCA space.
> "We gather data, multiply it using augmentation, and convert face images into 512 numbers. Same person clusters together. That cluster is the identity — not the photo."

### 2 · Model Building (90s)  → *"The model learns to draw boundaries"*
- Select **SVM**, **KNN**, or **MLP (Neural Network)** and click **🚀 Train model**.
- If **MLP** is chosen, show the **Iterative Training Dynamics** graphs (Loss Curve and Accuracy Progression) and the **Classifier Decision Space** boundary map.
> "Instead of just matching vectors, the model draws boundaries separating identities. With neural networks, we watch the loss drop epoch-by-epoch as the weights converge."

### 3 · Evaluation (90s)  → *"Predicting Unknown beats guessing"*
- Show **Accuracy / Precision / Recall / F1** and the **confusion matrix**.
- Drag the **threshold slider** and watch the **precision/recall curve** move.
> "Lower the bar and we accept more — but make more mistakes. Raise it and we sometimes say 'Unknown'. For attendance, a wrong name is worse than 'Unknown'."

### 4 · Deployment (75s)  → *"Where ML creates business value"*
- Pick **Browser camera**, capture yourself → watch **Identity + Confidence + Timestamp** fill in the attendance log.
- Try capturing twice to demonstrate the duplicate prevention check: `you have taken participation`.
> "Same pipeline, now live: Camera → Detection → Embedding → Search → Attendance. This is where ML creates business value."

---

### Close (15s)
> "Four stages, one pipeline. That's the core lifecycle of every machine-learning product."

**Tip:** use the sidebar **🔄 Reset demo data** button between audiences for a clean start.
