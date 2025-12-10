# Unified AI/Robotics Book

Welcome to the Unified AI/Robotics Book project - a comprehensive educational resource on Physical AI and Humanoid Robotics. This Docusaurus-based documentation site covers everything from ROS 2 fundamentals to advanced AI integration for humanoid robots.

## About This Book

This book is organized into 5 comprehensive modules:

1. **Module 1: The Robotic Nervous System (ROS 2)** - Learn ROS 2 fundamentals, Nodes, Topics, Services, and Python-ROS integration
2. **Module 2: The Digital Twin (Gazebo & Unity)** - Explore physics simulation and high-fidelity rendering
3. **Module 3: The AI-Robot Brain (NVIDIA Isaac)** - Dive into VSLAM, perception, and path planning
4. **Module 4: Vision-Language-Action (VLA)** - Combine vision, language, and action for autonomous behavior
5. **Module 5: Capstone - Autonomous Humanoid Robot** - Complete implementation of an autonomous humanoid robot

## Prerequisites

- Node.js version 18.0 or above
- npm or yarn package manager

## Installation

```bash
npm install
```

## Local Development

```bash
npm run start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

## Contributing

This educational resource is designed for students and developers learning Physical AI and humanoid robotics. Contributions are welcome, especially for:
- Technical accuracy improvements
- Additional examples and exercises
- Updated content based on latest ROS 2 and AI developments

## License

This educational content is provided for learning purposes in the field of robotics and AI.

## About This Project

This project was created as part of a comprehensive course on Physical AI and Humanoid Robotics, designed for intermediate-to-advanced students in robotics, AI engineering, and LLM systems.
