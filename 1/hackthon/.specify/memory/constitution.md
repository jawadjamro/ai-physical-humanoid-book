<!--
Sync Impact Report:
- Version change: N/A -> 1.0.0 (initial constitution)
- Modified principles: All principles added as initial content
- Added sections: All sections added as initial content
- Removed sections: None
- Templates requiring updates: ✅ .specify/templates/plan-template.md, ✅ .specify/templates/spec-template.md, ✅ .specify/templates/tasks-template.md
- Follow-up TODOs: None
-->
# Unified AI/Robotics Book + Integrated RAG Chatbot Constitution

## Core Principles

### Technical Accuracy and Documentation Integrity
All robotics concepts (ROS 2, Gazebo, Isaac, Nav2, VSLAM, URDF, humanoid control) must be referenced from primary/official sources; All AI sections (RAG, embeddings, Agents SDK, FastAPI, Qdrant, NeonDB) must follow real implementation guidelines; No hallucination: If information is unknown, request clarification instead of inventing details.

### Reproducibility and Educational Excellence
All tutorials, commands, and architectures must be testable by students; Writing style: Developer-friendly, instructional, and project-oriented; Each module must include practical examples and student exercises; All implementation steps reproducible on a student machine.

### Engineering Rigor and Open-Source Standards
Prioritize official documentation, open-source standards, and verified robotics workflows; All system diagrams must be consistent with ROS 2 / Gazebo / Isaac conventions; Follow real implementation guidelines from official sources; Maintain high code quality and engineering standards.

### Student-Centric Accessibility
Clarity for an audience of intermediate-to-advanced students in robotics, AI engineering, and LLM systems; Include diagrams, architecture flows, and interface specifications where needed; Support both learning and real implementation; Include a "Student Tasks" section for each module.

### Integration and Cross-Platform Compatibility
Backend stack: FastAPI + Neon Serverless Postgres + Qdrant Cloud Free Tier; Support vector search, metadata filtering, and document chunking; Ensure compatibility across ROS 2, Gazebo, Unity, NVIDIA Isaac, and OpenAI ecosystems; Enable seamless integration between AI and robotics components.

### Factual Precision and Implementation Reality
Must avoid speculative robotics claims (only cite factual, implementable methods); Content must be technically correct and aligned with official docs; Focus on implementable solutions rather than theoretical concepts; Verify all claims against official documentation before inclusion.

## Content Standards and Format Requirements

Book built using Docusaurus; Deployment: GitHub Pages with clean versioned documentation; Include API documentation for the RAG backend; Include code blocks for ROS 2, FastAPI, Python, and integration steps; RAG chatbot must answer questions based only on the book content and user-selected text (chunk-level references).

## Development Workflow and Quality Assurance

All robotics and AI explanations technically correct and aligned with official docs; Success criteria: Book builds successfully in Docusaurus and deploys without errors; RAG chatbot fully functional and retrieves accurate answers from book text; All robotics and AI explanations technically correct and aligned with official docs; Final deliverable supports both learning and real implementation.

## Governance

This constitution governs all aspects of the Unified AI/Robotics Book + Integrated RAG Chatbot project; All contributions must comply with these principles; Changes to core principles require explicit approval and documentation of impact; Regular reviews ensure continued alignment with educational and technical objectives.

**Version**: 1.0.0 | **Ratified**: 2025-12-09 | **Last Amended**: 2025-12-09
