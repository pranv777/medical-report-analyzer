from setuptools import setup, find_packages

setup(
    name="medical_rag",
    version="1.0.0",
    description="Medical Report Analyzer",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["src*"]),
    install_requires=[
        "pydantic>=2.5",
        "pydantic-settings>=2.1",
        "PyYAML>=6.0",
        "loguru>=0.7",
        "rich>=13.7",
        "typer>=0.9",
    ],
    entry_points={
        "console_scripts": [
            "medical-analyze=scripts.analyze_report:app",
            "medical-chat=scripts.chat:app",
            "medical-build-kb=scripts.build_knowledge_base:app",
            "medical-eval=scripts.run_evaluation:app",
        ]
    },
)
