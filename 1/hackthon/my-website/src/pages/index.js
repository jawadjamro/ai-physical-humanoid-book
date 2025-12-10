import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/module-1-ros2/intro">
            Start Reading the Book - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Unified AI/Robotics Book: Physical AI & Humanoid Robotics Educational Resource">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <section className={styles.additionalInfo}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <p style={{marginTop: '2rem', textAlign: 'center'}}>This comprehensive book covers everything from ROS 2 fundamentals to advanced AI integration for humanoid robotics. Perfect for intermediate-to-advanced students in robotics, AI engineering, and LLM systems.</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
