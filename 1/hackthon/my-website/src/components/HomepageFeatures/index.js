import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'ROS 2 Fundamentals',
    // Note: We'll need to add appropriate SVGs for robotics later
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Master the robotic nervous system that controls humanoid robots.
        Learn about Nodes, Topics, Services, and how Python-based Agents
        communicate with ROS 2 controllers using rclpy.
      </>
    ),
  },
  {
    title: 'Simulation & Digital Twins',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Explore Gazebo and Unity for physics simulation, sensors, collisions,
        and high-fidelity rendering. Create accurate digital twins of
        humanoid robots for testing and development.
      </>
    ),
  },
  {
    title: 'AI Integration',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Combine vision, language, and action for autonomous robot behavior.
        Learn Vision-Language-Action (VLA) systems, cognitive planning
        with LLMs, and ROS 2 action sequencing.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
