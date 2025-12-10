---
id: 1
title: Create Initial Constitution
stage: constitution
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 001-ros2-fundamentals
user: Claude
command: /sp.constitution
labels: [constitution,documentation,setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
/sp.constitution

Project: Unified AI/Robotics Book + Integrated RAG Chatbot (Physical AI & Humanoid Robotics)

Core Principles:
- Technical accuracy based on robotics, AI, and simulation documentation (ROS 2, Gazebo, Unity, NVIDIA Isaac, OpenAI, Qdrant, Postgres).
- Clarity for an audience of intermediate-to-advanced students in robotics, AI engineering, and LLM systems.
- Reproducibility: All tutorials, commands, and architectures must be testable by students.
- Engineering rigor: Prioritize official documentation, open-source standards, and verified robotics workflows.

Key Standards:
- All robotics concepts (ROS 2, Gazebo, Isaac, Nav2, VSLAM, URDF, humanoid control) must be referenced from primary/official sources.
- All AI sections (RAG, embeddings, Agents SDK, FastAPI, Qdrant, NeonDB) must follow real implementation guidelines.
- Writing style: Developer-friendly, instructional, and project-oriented.
- Include diagrams, architecture flows, and interface specifications where needed.
- Avoid hallucination: If information is unknown, request clarification instead of inventing details.

Format Requirements:
- Book built using Docusaurus.
- Deployment: GitHub Pages with clean versioned documentation.
- Include API documentation for the RAG backend.
- Include code blocks for ROS 2, FastAPI, Python, and integration steps.
- Include a "Student Tasks" section for each module.

RAG Chatbot Standards:
- Chatbot must answer questions based only on:
  1. The book content
  2. User-selected text (chunk-level references)
- Use OpenAI Agents/ChatKit SDKs.
- Backend stack: FastAPI + Neon Serverless Postgres + Qdrant Cloud Free Tier.
- Support vector search, metadata filtering, and document chunking.

Content Scope:
- Physical AI & Humanoid Robotics
- Embodied Intelligence
- Controlling humanoid robots in simulation and real-world settings

Modules Covered:
1. The Robotic Nervous System (ROS 2)
   - Nodes, Topics, Services, rclpy, URDF, humanoid joint modeling

2. The Digital Twin (Gazebo & Unity)
   - Physics simulation, sensors, collisions, high-fidelity rendering

3. The AI-Robot Brain (NVIDIA Isaac)
   - Synthetic data, VSLAM, Nav2, perception, path planning

4. Vision-Language-Action (VLA)
   - Whisper voice commands, cognitive planning via LLMs, ROS 2 action sequencing

5. Capstone: Autonomous Humanoid Robot
   - Receives voice command → plans → navigates → perceives object → manipulates it

Constraints:
- Book length: 12–20 chapters.
- Each module must include practical examples and student exercises.
- Must avoid speculative robotics claims (only cite factual, implementable methods).
- All system diagrams must be consistent with ROS 2 / Gazebo / Isaac conventions.

Success Criteria:
- Book builds successfully in Docusaurus and deploys without errors.
- RAG chatbot fully functional and retrieves accurate answers from book text.
- All implementation steps reproducible on a student machine.
- All robotics and AI explanations technically correct and aligned with official docs.
- Final deliverable supports both learning and real implementation.
```

## Response snapshot

Created a comprehensive project constitution for the Unified AI/Robotics Book + Integrated RAG Chatbot project. The constitution includes six core principles covering technical accuracy, reproducibility, engineering rigor, student-centric accessibility, integration compatibility, and factual precision. It also defines content standards, development workflow, and governance rules. The constitution has been saved to .specify/memory/constitution.md with version 1.0.0.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles and governance structure
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Plan development, spec creation, task breakdown
- 🧠 Reflection: Constitution establishes clear technical and educational standards for the AI/robotics book project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A