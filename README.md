# MBAI 417 : Data Intensive Systems
[[github link](https://github.com/drc-cs/SPRING25-DATA-INTENSIVE-SYSTEMS/tree/main)]

Welcome to Data Intensive Systems! This course provides a comprehensive introduction to distributed data processing and modeling.  These areas are essential to roles such as Data Engineers, Machine Learning Engineers, and Data Architects. The first half covers data-intensive operations including handling cloud storage, online analytical processing, and distributed data preprocessing. Building on this foundation, we transition towards data-intensive machine learning operations, including scalable machine learning algorithms (e.g. linear regression, XGBoost), distributed training, hyperparameter tuning, and model serving with real-time inference. These skills are essential for Machine Learning Engineers, Machine Learning Operations Engineers, and Technical Product Managers.

Assessments include two exams covering all material covered in the lectures and homeworks. Homework assignments focus on practical tasks processing large datasets and training machine learning models in the cloud, reinforcing learned concepts through real-world applications. These hands-on projects mirror the day-to-day work of industry professionals.

## Schedule

| Week | Date  | Lecture Title  | Topics  | Homework Due | Context for DIS | Location |
|------|-------|---------|-----|--------------------|----| ----|
| 1    | 04.04.2025 @ 3:30 PM | [Introduction](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L01/#/) | Course Overview; Environment Setup | ---  | ---  | Ford ITW Auditorium 1350 |
| 2    | 04.07.2025 @ 3:30 PM | [Databases](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L02/#/)  | Modern Database Landscape | ---  | Introduction to databases and their use cases. | Ford ITW Auditorium 1350 |
| 2    | 04.10.2025 @ 3:30 PM | [Databases II](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L03/#/) | Accessing Databases, Permissions, and Security | H.01  | Accessing databases and database security essentials | Ford ITW Auditorium 1350 |
| 3    | 04.14.2025 @ 3:30 PM | [Online Analytical Processing (OLAP)](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L04/#/) | OLTP vs OLAP, Better Schemas, Cubes, Columnar DBs | ---  | From OLTP to modern OLAP, analytics at scale. | Ford ITW Auditorium 1350 |
| 3   | 04.16.2025 @ 10:30 AM | [OLAP + EDA I](https://drc-cs.github.io/SPRING25-DATA-INTENSIVE-SYSTEMS/lectures/L05/#/) | Database Imputation & Outlier Detection Strategies | ---  | Handling and describing imperfect data. | Kellogg Global Hub L110 |
| 3   | 04.17.2025 @ 3:30 PM | OLAP + EDA II | Covariance, Correlation, Association Analysis | H.02  | Describing numerical patterns with online analytics. | Ford ITW Auditorium 1350 |
| 4   | 04.21.2025 @ 3:30 PM | OLAP + EDA III | Hypothesis Testing & A/B Testing | ---  | Describing numerical patterns with online analytics. | Ford ITW Auditorium 1350 |
| 4   | 04.24.2025 @ 3:30 PM | OLAP + EDA IV | Modern Text Mining (NLP) | ---  | Describing text patterns with online analytics. | Ford ITW Auditorium 1350 |
| 5   | 04.28.2025 @ 3:30 PM | Distributed Preprocessing I | Distributed Data Processing | --- | Leveraging cloud resources for data preprocessing. | Ford ITW Auditorium 1350 |
| 5    | 04.30.2025  @ 3:30 PM | Exam Review & Distributed Processing II | Exam Review and Feature Selection | ---  | Leveraging cloud resources for data preprocessing. | Ford ITW Auditorium 1350 |
| 6    | 05.05.2025  @ 3:30 PM | Exam Part I  | Assessment | H.03 | --- | Ford ITW Auditorium 1350 |
| 6    | 05.08.2025  @ 3:30 PM | Containerization | Creating, Managing, and Registering Containers | ---  | Utilizing containers in your workflow. | Ford ITW Auditorium 1350 |
| 7    | 05.12.2025 @ 3:30 PM | Scalable Machine Learning I | Regression and Classification Modeling | ---  | Regression modeling in practice. | Ford ITW Auditorium 1350 |
| 7    | 05.15.2025 @ 3:30 PM | Scalable Machine Learning II | Decision Trees, XGBoost | ---  | Non-linear modeling in practice. | Ford ITW Auditorium 1350 |
| 8    | 05.19.2025 @ 3:30 PM | Scalable Machine Learning III | Time Series Modeling (SARIMAX, TiDE) | H.04 | Generating forecasts from your database. | Ford ITW Auditorium 1350 |
| 8    | 05.21.2025 @ 10:30 AM | Modern NLP Applications | RAG Model & Text Generation | ---  | Generating company-specific text with RAG. | Kellogg Global Hub L110 |
| 8    | 05.22.2025 @ 3:30 PM | Unsupervised Machine Learning | Clustering Methods | ---  | Clustering unlabeled data. | Ford ITW Auditorium 1350 |
| 9    | 05.29.2025 @ 3:30 PM | MLOps | Hyperparameter Tuning Strategies & Distributed Training |  ---  | Training the best possible model. | Ford ITW Auditorium 1350 |
| 10    | 06.02.2025 @ 3:30 PM | Exam Review & Model Deployment | Model Serving & Real-Time Inference | H.05  | Deploying your model to production. | Ford ITW Auditorium 1350 |
| 10   | 06.05.2025 @ 3:30 PM | Exam Part II   | Assessment  |  ---  |  ---  | Ford ITW Auditorium 1350 |

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
