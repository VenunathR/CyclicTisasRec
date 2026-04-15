# CyclicTiSASRec: Project Overview

## 🚩 Where we are

We are building a sequential recommendation system based on self-attention models. Current transformer-based approaches like TiSASRec are effective at capturing item sequences and time intervals, but they mainly treat time as a linear progression.

In practice, user behavior is rarely purely linear. It often follows repeating patterns such as daily routines, weekly habits, or seasonal trends. These patterns are not fully captured by existing models.

---

## 🎯 What we are trying to do

The goal of this project is to extend time-aware recommendation by introducing **cyclic temporal modeling**.

Instead of only learning:

* what happens next in a sequence
* and how much time has passed

we also want the model to learn:

* what typically happens at similar *points in a cycle*

For example:

* user behavior at 9 PM across different days
* weekend vs weekday preferences

---

## 🧠 Core idea

We aim to enhance a self-attention based recommendation model by adding **cyclic time encoding**.

This involves:

* Representing time using periodic functions
* Capturing repeating behavioral patterns
* Combining this with sequence and interval awareness

The final model should better understand both:

* **sequence dynamics** (order of actions)
* **temporal dynamics** (when actions happen, including repetition)

---

## ⚙️ Current plan

1. Start with a baseline self-attention recommendation model
2. Add time interval encoding (as in TiSASRec)
3. Introduce cyclic time features (daily/weekly patterns)
4. Train and compare performance with baseline
5. Evaluate using ranking metrics

---

## 📌 Expected outcome

By incorporating cyclic temporal information, we expect:

* Improved recommendation quality
* Better modeling of long-term user behavior
* Stronger performance in time-sensitive applications

---

## 🚀 Next steps

* Implement cyclic encoding module
* Integrate it into attention pipeline
* Run experiments on benchmark datasets
* Analyze improvements over baseline

---

This project is an exploration into making recommendation systems more aligned with how real-world user behavior actually evolves over time.
