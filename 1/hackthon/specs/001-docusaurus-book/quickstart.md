# Quickstart: Docusaurus Book Development

## Prerequisites
- Node.js (LTS version recommended)
- npm or yarn package manager
- Git for version control
- Text editor or IDE with Markdown support

## Setup Instructions

### 1. Clone and Initialize
```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd <repository-name>

# Navigate to the website directory (create if it doesn't exist)
mkdir -p website
cd website
```

### 2. Install Docusaurus
```bash
# Create a new Docusaurus site
npm init docusaurus@latest . classic

# Or if you prefer yarn
yarn create docusaurus . classic
```

### 3. Project Structure Setup
```bash
# After Docusaurus installation, your structure should look like:
website/
├── blog/
├── docs/
├── src/
│   ├── components/
│   ├── css/
│   └── pages/
├── static/
├── docusaurus.config.js
├── sidebars.js
├── package.json
└── README.md
```

### 4. Install Additional Dependencies
```bash
# Navigate to website directory
cd website

# Install additional dependencies for enhanced functionality
npm install --save-dev @docusaurus/module-type-aliases @docusaurus/types
npm install @docusaurus/preset-classic @docusaurus/remark-plugin-npm2yarn
```

### 5. Configure Docusaurus
Update `docusaurus.config.js` with book-specific configuration:

```javascript
// docusaurus.config.js
module.exports = {
  title: 'Unified AI/Robotics Book',
  tagline: 'Physical AI & Humanoid Robotics Educational Resource',
  favicon: 'img/favicon.ico',

  url: 'https://your-username.github.io', // Replace with your URL
  baseUrl: '/unified-ai-robotics-book/', // Replace with your base URL

  organizationName: 'your-username', // Usually your GitHub org/user name
  projectName: 'unified-ai-robotics-book', // Usually your repo name

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/your-username/unified-ai-robotics-book/edit/main/website/',
        },
        blog: false, // Disable blog if not needed
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'AI/Robotics Book',
        logo: {
          alt: 'AI/Robotics Book Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book Modules',
          },
          {
            href: 'https://github.com/your-username/unified-ai-robotics-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/module-1-ros2/intro',
              },
              {
                label: 'Digital Twin',
                to: '/docs/module-2-gazebo/intro',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/your-username/unified-ai-robotics-book',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} AI/Robotics Educational Project. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer/themes/github'),
        darkTheme: require('prism-react-renderer/themes/dracula'),
        additionalLanguages: ['python', 'bash', 'json', 'yaml'],
      },
    }),
};
```

### 6. Create Module Structure
```bash
# Create module directories
mkdir -p docs/module-1-ros2
mkdir -p docs/module-2-gazebo
mkdir -p docs/module-3-isaac
mkdir -p docs/module-4-vla
mkdir -p docs/module-5-capstone
```

### 7. Create Sidebar Configuration
Update `sidebars.js` to organize content by modules:

```javascript
// sidebars.js
module.exports = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/intro',
        'module-1-ros2/chapter-1',
        'module-1-ros2/chapter-2',
        'module-1-ros2/chapter-3',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-gazebo/intro',
        'module-2-gazebo/chapter-1',
        'module-2-gazebo/chapter-2',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/intro',
        'module-3-isaac/chapter-1',
        'module-3-isaac/chapter-2',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/intro',
        'module-4-vla/chapter-1',
        'module-4-vla/chapter-2',
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Capstone - Autonomous Humanoid Robot',
      items: [
        'module-5-capstone/intro',
        'module-5-capstone/chapter-1',
      ],
    },
  ],
};
```

### 8. Development Workflow
```bash
# Start development server
cd website
npm start

# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

### 9. Content Creation Workflow
1. Create new markdown files in the appropriate module directory
2. Add frontmatter with title, description, and other metadata
3. Use Docusaurus markdown features (admonitions, tabs, etc.)
4. Update sidebars.js to include new content in navigation
5. Test locally before committing