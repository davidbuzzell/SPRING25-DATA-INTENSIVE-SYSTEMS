# MBAI 417 : Data Intensive Systems
[[github link](https://github.com/drc-cs/SPRING25-DATA-INTENSIVE-SYSTEMS/tree/main)]

Welcome to Data Intensive Systems! This course provides a comprehensive introduction to distributed data processing and modeling.  These areas are essential to roles such as Data Engineers, Machine Learning Engineers, and Data Architects. The first half covers data-intensive operations including handling cloud storage, online analytical processing, and distributed data preprocessing. Building on this foundation, we transition towards data-intensive machine learning operations, including scalable machine learning algorithms (e.g. linear regression, XGBoost), distributed training, hyperparameter tuning, and model serving with real-time inference. These skills are essential for Machine Learning Engineers, Machine Learning Operations Engineers, and Technical Product Managers.

Assessments include two exams covering all material covered in the lectures and homeworks. Homework assignments focus on practical tasks processing large datasets and training machine learning models in the cloud, reinforcing learned concepts through real-world applications. These hands-on projects mirror the day-to-day work of industry professionals.

## Schedule

| Week | Date  | Lecture Slides  | Printable Link | Topics  | Homework Due | Context for DIS | Location |
|------|-------|---------|-----|-----|--------------------|----| ----|
| 1    | 04.04.2025 @ 3:30 PM | [Introduction](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L01/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L01.pdf) | Course Overview; Environment Setup | ---  | ---  | Ford ITW Auditorium 1350 |
| 2    | 04.07.2025 @ 3:30 PM | [Databases](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L02/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L02.pdf) | Modern Database Landscape | ---  | Introduction to databases and their use cases. | Ford ITW Auditorium 1350 |
| 2    | 04.10.2025 @ 3:30 PM | [Databases II](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L03/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L03.pdf) | Accessing Databases, Permissions, and Security | H.01  | Accessing databases and database security essentials | Ford ITW Auditorium 1350 |
| 3    | 04.14.2025 @ 3:30 PM | [Online Analytical Processing (OLAP)](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L04/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L04.pdf)| OLTP vs OLAP, Better Schemas, Cubes, Columnar DBs | ---  | From OLTP to modern OLAP, analytics at scale. | Ford ITW Auditorium 1350 |
| 3   | 04.16.2025 @ 10:30 AM | [OLAP + EDA I](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L05/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L05.pdf)| Database Imputation & Outlier Detection Strategies | ---  | Handling and describing imperfect data. | Kellogg Global Hub L110 |
| 3   | 04.17.2025 @ 3:30 PM | [OLAP + EDA II](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L06/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L06.pdf)| Covariance, Correlation, Association Analysis | H.02  | Describing numerical patterns with online analytics. | Ford ITW Auditorium 1350 |
| 4   | 04.21.2025 @ 3:30 PM | [OLAP + EDA III](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L07/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L07.pdf)| Hypothesis Testing & A/B Testing | ---  | Describing numerical patterns with hypothesis testing. | Ford ITW Auditorium 1350 |
| 4   | 04.24.2025 @ 3:30 PM | [OLAP + EDA IV](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L08/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L08.pdf)| Modern Text Mining (NLP) | ---  | Describing text patterns with online analytics. | Ford ITW Auditorium 1350 |
| 5   | 04.28.2025 @ 3:30 PM | [Distributed Preprocessing I](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L09/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L09.pdf)| Distributed Data Processing | --- | Leveraging cloud resources for data preprocessing. | Ford ITW Auditorium 1350 |
| 5    | 04.31.2025  @ 3:30 PM | [Exam Review & Distributed Processing II](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L10/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L10.pdf) | Exam Review and Feature Selection | ---  | Leveraging cloud resources for data preprocessing. | Ford ITW Auditorium 1350 |
| 6    | 05.05.2025  @ 3:30 PM | Exam Part I  | | Assessment | H.03 | --- | Ford ITW Auditorium 1350 |
| 6    | 05.08.2025  @ 3:30 PM | [Containerization](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L11/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L11.pdf) | Creating, Managing, and Registering Containers | ---  | Utilizing containers in your workflow. | Ford ITW Auditorium 1350 |
| 7    | 05.12.2025 @ 3:30 PM | [Scalable Machine Learning I](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L12/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L12.pdf) | Regression and Classification Modeling | ---  | Regression modeling in practice. | Ford ITW Auditorium 1350 |
| 7    | 05.15.2025 @ 3:30 PM | [Scalable Machine Learning II](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L13/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L13.pdf) | Decision Trees, XGBoost | ---  | Non-linear modeling in practice. | Kellogg Global Hub 2130 |
| 8    | 05.19.2025 @ 3:30 PM | [Scalable Machine Learning III](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L14/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L14.pdf) | Time Series Modeling | --- | Generating forecasts from your database. | Ford ITW Auditorium 1350 |
| 8    | 05.21.2025 @ 10:30 AM | [Modern NLP Applications](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L15/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L15.pdf) | RAG Model & Text Generation w/ MCP | H.04 | Generating company-specific text with RAG. | Kellogg Global Hub L110 |
| 8    | 05.22.2025 @ 3:30 PM | [Unsupervised Machine Learning](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L16/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L16.pdf) | Clustering Methods. | ---  | Clustering unlabeled data. | Ford ITW Auditorium 1350 |
| 9    | 05.29.2025 @ 3:30 PM | [Training the Best Model](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L17/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L17.pdf) | Hyperparameter Tuning Strategies |  ---  | Training the best possible model. | Ford ITW Auditorium 1350 |
| 10    | 06.02.2025 @ 3:30 PM | [Exam Review & Model Deployment](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L18/#/) | [[printable_link]](https://storage.googleapis.com/slide_assets/Lectures/L18.pdf) | Model Serving & Real-Time Inference | -- | Deploying your model to production. | Ford ITW Auditorium 1350 |
| 10   | 06.05.2025 @ 3:30 PM | Exam Part II  | | Assessment  |  ---  |  ---  | Ford ITW Auditorium 1350 |

## Homework Assignments

| Assignment | Due Date | Description | Details |
| --- | --- | --- | --- |
| H.01 | -- | Environment Setup | Setting up your environment for the course |
| H.02 | -- | SQL Basics | Building familiarity with SQL |
| H.03 | -- | Distributed Preprocessing | Building an ML-ready dataset |
| H.04 | -- | Machine Learning I | Training and evaluating models |
| H.05 | -- | Machine Learning II | Training models at scale |

## Required Software

### Docker [[download]](https://www.docker.com/products/docker-desktop)

Please have docker installed on your computer. You will perform your homework assignments in the provided docker, and we will go through the setup process on the first day of class.

### Chrome [[download]](https://www.google.com/chrome/)

Please have the latest version of Chrome installed on your computer. We will be using the browser for online polling and live Q/A during lectures, and Chrome has some additional features that are necessary to run the online system.

## Grading
 
| Component | Weight |
| --- | --- |
| Attendance & Comprehension | 20% | 
| Homework Assignments | 50% |
| Exam Part I | 15% |
| Exam Part II | 15% |
------

## Grading Scale

| Grade | Percentage |
| --- | --- |
| A | 93-100 |
| A- | 90-92 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |

------

## Attendance

Attendance at lectures is mandatory and in your best interest. 

Your **Attendance & Comprehension** score is worth 20% of your final grade. Lectures will have graded quizzes throughout, and the top 16 scores will be used to calculate your final grade.

------

## Lecture Breakdown

Lectures will be broken down into theory and application time. Every lecture will have online polling for live Q/A and comprehension checks. 

| Component | Duration | Description |
| --- | --- | --- |
| Lecture | 60 minutes | Theoretical Foundations |
| Demonstration Time | 20 minutes | Application and Practice |

------

## Academic Integrity [[link]](https://www.northwestern.edu/provost/policies-procedures/academic-integrity/index.html)

Individual homeworks are to be completed independently. You are encouraged to discuss the material with your peers, but the work you submit should be your own.

------

## Accommodations [[link]](https://www.registrar.northwestern.edu/registration-graduation/northwestern-university-syllabus-standards.html#accessibility)

Any student requesting accommodations related to a disability or other condition is required to register with AccessibleNU and provide professors with an accommodation notification from AccessibleNU, preferably within the first two weeks of class. 

All information will remain confidential.

------

## Mental Health [[link]](https://www.registrar.northwestern.edu/registration-graduation/northwestern-university-syllabus-standards.html#wellness-and-health)

If you are feeling distressed or overwhelmed, please reach out for help. Students can access confidential resources through the Counseling and Psychological Services (CAPS), Religious and Spiritual Life (RSL) and the Center for Awareness, Response and Education (CARE).

-----

**Office Hours**: 
- Thursdays (D'Arcy) from 2:00 PM to 3:00 PM in Mudd 3510, or by appointment.
- Fridays (Mo) from 3:00 PM to 4:00 PM in Mudd First Floor Lobby. 

-----

## Contact Information

Nathan Mo (Peer Mentor): NathanMo2026@u.northwestern.edu<br>
Joshua D'Arcy (Instructor): joshua.darcy@northwestern.edu
