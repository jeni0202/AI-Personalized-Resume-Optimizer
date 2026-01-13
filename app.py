import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import tempfile
import os
import sys

# Add current directory to Python path to ensure imports work in deployment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from parsers.document_parser import DocumentParser
from nlp.skill_extractor import SkillExtractor
from comparison.similarity_comparator import SimilarityComparator

# Set page configuration
st.set_page_config(
    page_title="AI Resume Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        margin: 0.125rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎯 AI Personalized Resume Optimizer</h1>', unsafe_allow_html=True)

    # Initialize components
    @st.cache_resource
    def load_components():
        parser = DocumentParser()
        extractor = SkillExtractor()
        comparator = SimilarityComparator()
        return parser, extractor, comparator

    parser, extractor, comparator = load_components()

    # Sidebar for file uploads
    st.sidebar.header("📁 Upload Documents")

    # Resume upload
    resume_file = st.sidebar.file_uploader(
        "Upload Resume (PDF/DOCX)",
        type=['pdf', 'docx'],
        help="Upload your resume in PDF or DOCX format"
    )

    # Job description upload
    jd_file = st.sidebar.file_uploader(
        "Upload Job Description (PDF/DOCX/TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload the job description you want to match against"
    )

    # Text input alternative for JD
    jd_text = st.sidebar.text_area(
        "Or paste Job Description text",
        height=100,
        help="Alternatively, paste the job description text directly"
    )

    # Main content area
    if resume_file or jd_file or jd_text:
        col1, col2 = st.columns(2)

        resume_text = ""
        jd_content = ""

        # Process resume
        if resume_file:
            with col1:
                st.subheader("📄 Resume Analysis")

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(resume_file.read())
                    tmp_path = tmp_file.name

                try:
                    resume_text = parser.parse_document(tmp_path)
                    st.success("✅ Resume parsed successfully!")

                    # Extract skills
                    resume_skills = extractor.extract_skills(resume_text)
                    resume_skill_categories = extractor.categorize_skills(resume_skills)

                    # Display skills
                    st.write(f"**Extracted Skills ({len(resume_skills)}):**")
                    skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in resume_skills[:20]])
                    if len(resume_skills) > 20:
                        skill_html += f'<span class="skill-tag">+{len(resume_skills)-20} more</span>'
                    st.markdown(skill_html, unsafe_allow_html=True)

                    # Skills by category
                    if resume_skill_categories:
                        st.subheader("Skills by Category")
                        for category, skills in resume_skill_categories.items():
                            with st.expander(f"{category.title()} ({len(skills)})"):
                                st.write(", ".join(skills))

                except Exception as e:
                    st.error(f"Error parsing resume: {str(e)}")
                finally:
                    os.unlink(tmp_path)

        # Process job description
        if jd_file or jd_text:
            with col2:
                st.subheader("💼 Job Description Analysis")

                if jd_file:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(jd_file.read())
                        tmp_path = tmp_file.name

                    try:
                        jd_content = parser.parse_document(tmp_path)
                    except Exception as e:
                        st.error(f"Error parsing job description: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
                else:
                    jd_content = jd_text

                if jd_content:
                    st.success("✅ Job description processed!")

                    # Extract skills
                    jd_skills = extractor.extract_skills(jd_content)
                    jd_skill_categories = extractor.categorize_skills(jd_skills)

                    # Display skills
                    st.write(f"**Required Skills ({len(jd_skills)}):**")
                    skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in jd_skills[:20]])
                    if len(jd_skills) > 20:
                        skill_html += f'<span class="skill-tag">+{len(jd_skills)-20} more</span>'
                    st.markdown(skill_html, unsafe_allow_html=True)

                    # Skills by category
                    if jd_skill_categories:
                        st.subheader("Required Skills by Category")
                        for category, skills in jd_skill_categories.items():
                            with st.expander(f"{category.title()} ({len(skills)})"):
                                st.write(", ".join(skills))

        # Comparison section
        if resume_text and jd_content:
            st.header("🔍 Resume-Job Match Analysis")

            # Calculate similarity
            try:
                similarity_results = comparator.compare_resume_to_jd(resume_text, jd_content)

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Overall Match",
                        f"{similarity_results['overall_similarity']:.1%}",
                        help="Semantic similarity between resume and job description"
                    )

                with col2:
                    st.metric(
                        "Avg Section Match",
                        f"{similarity_results['avg_section_similarity']:.1%}",
                        help="Average similarity across document sections"
                    )

                with col3:
                    st.metric(
                        "Resume Skills",
                        len(resume_skills),
                        help="Number of skills extracted from resume"
                    )

                with col4:
                    st.metric(
                        "Required Skills",
                        len(jd_skills),
                        help="Number of skills extracted from job description"
                    )

                # Skills comparison
                st.subheader("🎯 Skills Match Analysis")

                # Find matching and missing skills
                matching_skills = set(resume_skills) & set(jd_skills)
                missing_skills = set(jd_skills) - set(resume_skills)
                extra_skills = set(resume_skills) - set(jd_skills)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("✅ Matching Skills")
                    st.write(f"**{len(matching_skills)} skills found**")
                    if matching_skills:
                        skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in list(matching_skills)[:10]])
                        if len(matching_skills) > 10:
                            skill_html += f'<span class="skill-tag">+{len(matching_skills)-10} more</span>'
                        st.markdown(skill_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("📈 Skills to Develop")
                    st.write(f"**{len(missing_skills)} skills needed**")
                    if missing_skills:
                        skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in list(missing_skills)[:10]])
                        if len(missing_skills) > 10:
                            skill_html += f'<span class="skill-tag">+{len(missing_skills)-10} more</span>'
                        st.markdown(skill_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("🔧 Additional Skills")
                    st.write(f"**{len(extra_skills)} extra skills**")
                    if extra_skills:
                        skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in list(extra_skills)[:10]])
                        if len(extra_skills) > 10:
                            skill_html += f'<span class="skill-tag">+{len(extra_skills)-10} more</span>'
                        st.markdown(skill_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Visualization
                st.subheader("📊 Skills Distribution")

                # Create comparison chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Skills by category comparison
                resume_cats = list(resume_skill_categories.keys())
                resume_counts = [len(resume_skill_categories[cat]) for cat in resume_cats]
                jd_cats = list(jd_skill_categories.keys())
                jd_counts = [len(jd_skill_categories[cat]) for cat in jd_cats]

                # Combine categories
                all_cats = list(set(resume_cats + jd_cats))
                resume_vals = [len(resume_skill_categories.get(cat, [])) for cat in all_cats]
                jd_vals = [len(jd_skill_categories.get(cat, [])) for cat in all_cats]

                x = range(len(all_cats))
                ax1.bar([i - 0.2 for i in x], resume_vals, 0.4, label='Resume', alpha=0.8)
                ax1.bar([i + 0.2 for i in x], jd_vals, 0.4, label='Job Description', alpha=0.8)
                ax1.set_xticks(x)
                ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in all_cats], rotation=45, ha='right')
                ax1.set_ylabel('Number of Skills')
                ax1.set_title('Skills by Category')
                ax1.legend()

                # Match quality gauge
                similarity_score = similarity_results['overall_similarity']
                colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
                color_idx = min(int(similarity_score * 5), 4)

                ax2.pie([similarity_score, 1-similarity_score],
                       colors=[colors[color_idx], 'lightgray'],
                       startangle=90,
                       counterclock=False)
                ax2.set_title(f'Overall Match Quality\n{similarity_score:.1%}')

                plt.tight_layout()
                st.pyplot(fig)

                # Recommendations
                st.subheader("💡 Recommendations")

                match_quality = comparator.get_similarity_score(similarity_score)

                if similarity_score >= 0.8:
                    st.success("🎉 Excellent match! Your resume is well-aligned with this job description.")
                    st.info("Consider tailoring your resume further by emphasizing the matching skills in your experience section.")
                elif similarity_score >= 0.6:
                    st.info("👍 Good match! Your resume has strong alignment with the job requirements.")
                    if missing_skills:
                        st.warning(f"Consider highlighting or acquiring these skills: {', '.join(list(missing_skills)[:5])}")
                elif similarity_score >= 0.4:
                    st.warning("⚠️ Fair match. Your resume partially matches the job requirements.")
                    if missing_skills:
                        st.error(f"Focus on developing these key skills: {', '.join(list(missing_skills)[:5])}")
                else:
                    st.error("❌ Poor match. Significant gaps exist between your resume and job requirements.")
                    if missing_skills:
                        st.error(f"Priority skills to develop: {', '.join(list(missing_skills)[:5])}")

            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")

    else:
        # Welcome message when no files uploaded
        st.info("👋 Welcome to the AI Resume Optimizer!")
        st.markdown("""
        **How to use this tool:**

        1. **Upload your resume** (PDF or DOCX format) in the sidebar
        2. **Upload or paste a job description** you want to match against
        3. **Get instant analysis** including:
           - Skill extraction from both documents
           - Semantic similarity scoring
           - Skills gap analysis
           - Personalized recommendations

        **Features:**
        - 🤖 AI-powered skill extraction using NLP
        - 📊 Semantic matching with Sentence-BERT
        - 📈 Visual skill comparisons
        - 💡 Actionable recommendations
        """)

        # Sample data for demonstration
        st.subheader("🔍 Try with Sample Data")
        if st.button("Load Sample Resume & Job Description"):
            sample_resume = """
            Software Engineer with 5 years of experience in Python, Java, and web development.
            Proficient in Django, React, and AWS. Strong background in data structures and algorithms.
            Experience with agile methodologies and team collaboration.
            """

            sample_jd = """
            We are looking for a Software Engineer with experience in Python, JavaScript, and cloud technologies.
            Knowledge of React, Django, and AWS is a plus. Familiarity with agile methodologies required.
            Strong communication and problem-solving skills essential.
            """

            st.session_state.sample_resume = sample_resume
            st.session_state.sample_jd = sample_jd
            st.rerun()

if __name__ == "__main__":
    main()
