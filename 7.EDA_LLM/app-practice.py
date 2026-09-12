
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

# Function to perform EDA and Generate Visualization
def eda_analysis(file_path):
    df = pd.read_csv(file_path)
    
    # Fill missing values with median for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        #df[col].fillna(df[col].median(),inplace=True)
        df[col] = df[col].fillna(df[col].median())
            
    # Fill missing values with mode for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        #df[col].fillna(df[col].mode()[0],inplace=True)
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Data Summary
    summary = df.describe(include='all').to_string()
    
    #Missing values
    missing_values = df.isnull().sum().to_string()
    
    # Generate Ai insights
    insights = generate_ai_insights(summary)
    
    #Generate Data Visualization
    plot_paths = generate_ai_visualization(df)
    
    return f"\n Data Loaded Successfully!\n\n Summary:\n{summary}\n\n Missing Values:\n{missing_values}\n\n AI Insights:\n{insights}",plot_paths

# AI -Powered Insights using gemma3:4b ollama
def generate_ai_insights(df_summary):
    prompt = f"Analyze the dataset summary and provide insights:\n\n{df_summary}"
    response = ollama.chat(model='gemma3:4b',messages=[{"role":"user","content":prompt}])
    return response['message']['content']

# Function to generate Data visualizations
def generate_ai_visualization(df):
    plot_paths = []
    # Histograms for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col],bins=30,kde=True,color="blue")
        plt.title(f"Distribution of {col}")
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
    
    
    # Correlation Heatmap for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8,5))
        sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt=".2f",linewidths=0.5)
        plt.title("Correlation Heatmap")
        path = "correlation_heatmap.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
    return plot_paths
    
# Gradio Interface
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath"),
    outputs=[gr.Textbox(label="EDA Report"),gr.Gallery(label="Data Visualizations")],
    title="LLM-Powered Exploratory Data Analysis",
    description="Upload any dataset CSV file and get automated EDA insights with AI Powered analysis and visualyzation"
)

#Launch the Gradio APP
demo.launch(share=True)
