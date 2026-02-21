# FloodAI
AI-driven flood risk intelligence system providing property-level hazard analysis using NLP, GIS, and pre-simulated flood models for urban decision support.
🌊 FloodAI
An AI-Driven Flood Risk Intelligence Agent for Property-Level Urban Decision Support

FloodAI is an applied AI + GIS decision-support system that delivers property-level flood risk analysis using natural language input, official rainfall design data, and pre-computed flood simulation maps.

The system is designed to make complex flood modelling outputs accessible to non-experts (homeowners, councils, emergency planners) through explainable, real-time insights.

🧠 What Problem Does FloodAI Solve?

Urban flood risk information is typically:

Locked inside technical flood studies

Hard to interpret without hydrology/GIS expertise

Not accessible in real time during heavy rainfall events

FloodAI bridges this gap by allowing users to ask simple questions like:

“70 mm of rain in 1 hour at Frankston South — will my property flood?”

…and receive:

Flood depth estimates

Hazard classification (Low → Extreme)

Plain-language explanations

Visual flood maps

—all in seconds.

✨ Key Features

🗣️ Natural Language Input

Parses free-text rainfall and location descriptions using NLP (spaCy + regex)

🌧️ Rainfall Severity Classification

Maps rainfall to IFD / ARI using Bureau of Meteorology design rainfall data

🗺️ GIS-Based Flood Intelligence

Queries pre-simulated flood depth & hazard rasters stored in PostGIS

⚠️ Explainable Hazard Assessment

Converts depth & velocity into human-readable hazard levels (Low / Moderate / High / Extreme)

📊 Interactive Visual Output

Streamlit dashboard with flood maps and downloadable reports

🏙️ Property-Level Focus

Designed for individual addresses, not just suburb-level summaries

🏗️ System Architecture (High-Level)
User Query (Text)
        ↓
NLP Parsing (spaCy + Regex)
        ↓
Rainfall → ARI Mapping (BoM IFD Data)
        ↓
Flood Scenario Selection
        ↓
PostGIS Raster Query (Depth / Hazard)
        ↓
Hazard Classification & Explanation
        ↓
Streamlit Map + Text Report


This architecture enables near real-time responses by querying pre-computed flood simulations instead of running hydraulic models on demand.

🧪 Case Study: Frankston City Council (Victoria, Australia)

FloodAI was implemented and tested using real flood model data provided for the City of Frankston.

Example Use Cases:

Rapid assessment during extreme rainfall

“What-if” planning scenarios for infrastructure upgrades

Identifying high-risk properties within the same suburb

Supporting emergency response and communication

Observed Results:

Flood depth and hazard classifications aligned with known flood-prone locations

Correct differentiation between low-lying and elevated properties

Effective communication of risk to non-technical users

🧰 Tech Stack

Programming & Data

Python

Pandas, NumPy

NLP

spaCy

Regular Expressions

Fuzzy matching (location correction)

GIS & Spatial Analysis

PostgreSQL + PostGIS

Raster flood depth & hazard grids

GeoPandas, Rasterio

Standards & Domain Data

Bureau of Meteorology IFD (ARR 2016)

Australian flood hazard guidelines (depth-velocity thresholds)

Interface

Streamlit

Interactive maps & report generation

⚠️ Limitations (Current Version)

Uses static flood scenarios (e.g., 10-yr, 50-yr, 100-yr ARI)

Does not yet ingest live rainfall or sensor data

Geographic scope limited to Frankston case study

Not intended to replace official emergency warnings

These constraints are intentional to prioritise speed, reliability, and explainability.

🔮 Future Extensions

Planned or proposed enhancements:

Real-time rainfall ingestion (BoM radar / API)

Dynamic rainfall-runoff modelling

IoT drainage & water-level sensor integration

Climate-change uplift scenarios

Expansion to other councils and regions

Conversational chatbot interface


📄 Academic Context

This project was developed as part of an applied AI and smart-cities research initiative and is suitable for:

Final-year / capstone projects

Smart city decision-support systems

Urban resilience & climate adaptation research

📌 Disclaimer

FloodAI provides model-based predictive insights only.
Users should always follow official warnings from emergency services and meteorological authorities.
