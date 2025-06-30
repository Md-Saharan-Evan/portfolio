from django.shortcuts import render


def home(request):
    context = {
        'summary': "Passionate AI/ML researcher with expertise in deep learning, computer vision, and natural language processing. Proficient in Python, TensorFlow, PyTorch, and advanced ML libraries. Strong problem-solving, research, and collaboration skills, with a focus on developing innovative AI solutions for real-world applications.",
        'technical_skills': {
            'Programming Languages': ['Python', 'C', 'Java', 'JavaScript'],
            'Deep Learning Frameworks': ['TensorFlow', 'PyTorch', 'Keras', 'Caffe'],
            'Libraries & Tools': ['NumPy', 'Pandas', 'Scikit-learn', 'OpenCV', 'NLTK', 'Tesseract OCR', 'YOLOv8', 'Git', 'Docker', 'Tableau', 'Power BI', 'SQL'],
            'Web Development': ['Django', 'HTML5', 'CSS', 'JavaScript']
        },
        'experience': [
            {
                'title': 'AI/ML Intern',
                'company': 'Teton Bangladesh, Badda, Dhaka',
                'duration': 'Nov 2024 - Present',
                'responsibilities': [
                    'Evaluated and selected optimal models for product development',
                    'Designed customized models for multiclass object detection',
                    'Performed dataset analysis to enhance model performance'
                ]
            },
            {
                'title': 'AI Researcher',
                'company': 'Datalytics Labs, Uttara, Dhaka',
                'duration': 'Aug 2024 - Sep 2024',
                'responsibilities': [
                    'Conducted research on computer vision models',
                    'Developed object detection models tailored to specific use cases',
                    'Participated in daily scrums to report progress'
                ]
            },
            {
                'title': 'Assistant Instructor',
                'company': 'Glorious Professional Academy, Greenroad, Dhaka',
                'duration': 'Jan 2020 - Dec 2020',
                'responsibilities': [
                    'Delivered classes on Mathematics and Physics',
                    'Created logical reasoning question papers for assessments'
                ]
            }
        ],
        'education': [
            {
                'institution': 'BRAC University, Middle Badda, Dhaka',
                'degree': 'BSc in Computer Science and Engineering',
                'cgpa': '3.58',
                'duration': 'Aug 2020 - Dec 2024'
            },
            {
                'institution': 'Milestone College, Uttara, Dhaka',
                'degree': 'Science Group',
                'cgpa': '5.00',
                'duration': 'Jul 2017 - May 2019'
            }
        ],
        'research': [
            {
                'title': 'Safe RL-based System for Connected and Autonomous Vehicle Charging Infrastructure',
                'type': 'Undergraduate Thesis',
                'date': 'Mar 2024',
                'description': 'Developed an end-to-end system for optimal charging recommendations using Safe Reinforcement Learning, leveraging Deep Neural Networks, Reinforcement Learning, and Multiple Linear Models for enhanced performance.',
                'tools': ['Python', 'PyTorch']
            },
            {
                'title': 'Evaluating Online Sexism Detection',
                'type': 'Natural Language Processing',
                'date': 'Jan 2023 - Jan 2024',
                'description': 'Performed a comparative study of machine learning models on the EDOS dataset for sexism detection, implementing and evaluating multiple models.',
                'tools': ['Python', 'NLTK', 'Keras'],
                'link': 'https://ieeexplore.ieee.org/document/10543680/'
            }
        ],
        'projects': [
            {
                'title': 'NID Detection & Text Extraction Pipeline',
                'type': 'Deep Learning Project',
                'date': 'May 2025',
                'description': 'Built a deep learning pipeline using YOLOv8 for National ID card detection and Tesseract OCR for extracting structured information (Name, NID Number, Date of Birth, Address). Included image preprocessing (denoising, thresholding) and regex-based JSON output.',
                'tools': ['Python', 'YOLOv8', 'Tesseract OCR', 'OpenCV'],
                'images': [
                    '/static/images/nid_detection_1.jpeg',
                    '/static/images/nid_detection_2.jpeg',
                    '/static/images/nid_detection_3.jpeg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/nid-detection-demo'
            },
            {
                'title': 'Hospital Management System',
                'type': 'Web Development Project',
                'date': 'Feb 2023 - Apr 2023',
                'description': 'Developed a Django-based centralized server for hospitals and doctors, enabling patients to view specialists and medical records, and doctors to submit prescriptions and refer patients to other hospitals.',
                'tools': ['Django', 'HTML5', 'CSS', 'JavaScript'],
                'images': [
                    '/static/images/hospital_management_1.jpg',
                    '/static/images/hospital_management_2.jpg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/Hospital-Management-System-using-Django'
            },
            {
                'title': 'Image Captioning with BLIP',
                'type': 'Vision Language Model Project',
                'date': '2024',
                'description': 'Utilized the pretrained BLIP model for efficient image caption generation, achieving high performance with minimal setup time. Ideal for projects requiring auto-captioning features.',
                'tools': ['Python', 'BLIP', 'Deep Neural Networks'],
                'images': [
                    '/static/images/image_captioning_1.jpeg',
                    '/static/images/image_captioning_2.jpeg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/Image-Captioning-with-BLIP'
            },
            {
                'title': 'Signboard Detection from Roadside Images',
                'type': 'Deep Learning Project',
                'date': 'Feb 2024 - Present',
                'description': 'Developed a system for detecting signboards using CNN, RCNN, and YOLO models, with custom dataset annotation for training.',
                'tools': ['Python', 'TensorFlow', 'OpenCV'],
                'images': [
                    '/static/images/signboard_detection_1.jpeg',
                    '/static/images/signboard_detection_2.jpeg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/signboard-detection-demo'
            },
            {
                'title': 'Error Detection in Bengali Language',
                'type': 'Natural Language Processing',
                'date': 'Aug 2022 - Dec 2022',
                'description': 'Applied SVM and RF models to detect errors in Bengali text, using a Kaggle dataset with preprocessing.',
                'tools': ['Python', 'NLTK'],
                'images': [
                    '/static/images/bengali_error_detection_1.jpg',
                    '/static/images/bengali_error_detection_2.jpg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/bengali-error-detection-demo'
            },
            {
                'title': 'House Price Prediction',
                'type': 'Machine Learning',
                'date': 'Apr 2023 - Dec 2023',
                'description': 'Used tree-based models to predict house prices, achieving high accuracy with ensemble methods after data preprocessing.',
                'tools': ['Python', 'Scikit-learn'],
                'images': [
                    '/static/images/house_price_prediction_1.jpg'
                ],
                'github': 'https://github.com/Md-Saharan-Evan/house-price-prediction-demo'
            }
        ],
        'certifications': [
            'Software Engineering Virtual Experience - 2023, JP Morgan Chase Co. - Forage',
            'Data Science Virtual Experience Programme - 2023, British Airways'
        ],
        'achievements': [
            'Runner-up of TechSpectra 2.0, BRAC University, Dec 2023, Prompt Engineering, ChatGPT, Web Application',
            'Solved 100+ Problems on Leetcode: https://leetcode.com/u/Md_Saharan_Evan/',
            'Solved 50+ Problems on Codeforces: https://codeforces.com/profile/Saharan_Evan/'
        ],
        'references': [
            {
                'name': 'Dr. Md. Golam Rabiul Alam',
                'position': 'Professor, Dept. of Computer Science and Engineering, BRAC University',
                'contact': 'Email: rabiul.alam@bracu.ac.bd, Tel: +8809617445063, Mobile: +8801797347635',
                'website': 'https://www.bracu.ac.bd/about/people/md-golam-rabiul-alam-phd'
            },
            {
                'name': 'Dr. Md. Ashraful Alam',
                'position': 'Professor, Dept. of Computer Science and Engineering, BRAC University',
                'contact': 'Email: ashraful.alam@bracu.ac.bd',
                'website': 'https://www.bracu.ac.bd/about/people/md-ashraful-alam-phd'
            }
        ],
        'cv_file': '/media/Md_Saharan_Evan_research.pdf'
    }
    return render(request, 'portfolio/home.html', context)