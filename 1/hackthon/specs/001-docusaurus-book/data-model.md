# Data Model: Docusaurus Book Structure

## Content Entities

### Book
- **name**: Unified AI/Robotics Book
- **description**: Comprehensive educational resource covering ROS 2, Gazebo, NVIDIA Isaac, and AI integration for humanoid robotics
- **modules**: Array of Module entities (5 total)
- **target_audience**: Intermediate-to-advanced students in robotics, AI engineering, and LLM systems
- **word_count_range**: 1200-1800 words per chapter

### Module
- **id**: Unique identifier (e.g., "module-1-ros2")
- **title**: Descriptive title (e.g., "The Robotic Nervous System (ROS 2)")
- **description**: Brief overview of module content
- **chapters**: Array of Chapter entities (2-3 per module)
- **learning_objectives**: Array of specific learning goals
- **prerequisites**: Array of required knowledge or skills

### Chapter
- **id**: Unique identifier (e.g., "chapter-1-fundamentals")
- **title**: Descriptive title
- **content**: Markdown content with frontmatter
- **word_count**: Actual word count (within 1200-1800 range)
- **difficulty_level**: Beginner/Intermediate/Advanced
- **estimated_reading_time**: Calculated based on word count
- **related_topics**: Array of related topics or chapters

### ContentElement
- **type**: text, code_block, diagram, image, table, etc.
- **content**: The actual content
- **caption**: Optional caption for figures/tables
- **alt_text**: For accessibility (images)
- **language**: For code blocks (python, cpp, bash, etc.)

## Navigation Structure

### Sidebar
- **module_nav**: Hierarchical navigation by modules and chapters
- **sidebar_items**: Array of navigation items with titles and paths
- **previous_next**: Links to previous and next chapters

### Metadata
- **frontmatter**: Title, description, tags, authors, date created
- **tags**: Array of relevant tags for search and categorization
- **authors**: Array of content authors
- **reviewers**: Array of content reviewers

## Relationships
- Book contains multiple Modules (1 to many)
- Module contains multiple Chapters (1 to many)
- Chapter contains multiple ContentElements (1 to many)
- ContentElements may reference other ContentElements (cross-references)