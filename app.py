import streamlit as st
import pandas as pd
from Text_Summarizer import process_input
import subprocess
from io import BytesIO
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_poppler_installed():
    """Check if Poppler is installed and accessible."""
    try:
        result = subprocess.run(
            ["pdftoppm", "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "pdftoppm" in result.stderr or "pdftoppm" in result.stdout:
            print("✅ Poppler is installed and accessible.")
            return True
        else:
            print("❌ Poppler is NOT installed or not in PATH.")
            return False
    except FileNotFoundError:
        print("❌ Poppler is NOT installed or not in PATH.")
        return False

check_poppler_installed()

def format_highlights(highlights_text):
    """Convert plain text highlights to formatted markdown for display and remove Markdown for saving."""
    if "•" in highlights_text or "-" in highlights_text:
        formatted_text = highlights_text.replace("•", "-").replace("—", "--")
    else:
        formatted_text = highlights_text
    
    # Remove Markdown bold formatting before saving to Excel
    formatted_text = re.sub(r'\*\*(.*?)\*\*', r'\1', formatted_text)  # Removes **bold** formatting
    
    return formatted_text

def main():
    st.title("Stateside Bill Summarization")
    st.write("Enter a URL of Stateside bill to summarize its content or upload an Excel file.")

    # Initialize session state for persistent data
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'prev_upload' not in st.session_state:
        st.session_state.prev_upload = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'all_summaries' not in st.session_state:
        st.session_state.all_summaries = {}

    # Individual URL processing
    url = st.text_input("Enter URL")
    if st.button("Summarize URL"):
        if url:
            with st.spinner('Processing...'):
                result = process_input(url)
                if isinstance(result, dict) and "error" in result:
                    st.warning(f"Failed to process URL: {result['error']}")
                elif isinstance(result, dict):
                    st.success("Summarization complete!") 
                    st.session_state.all_summaries[url] = {
                        "abstractive": result["abstractive"],
                        "extractive": result["extractive"],
                        "highlights": format_highlights(result["highlights"])
                    }
                else:
                    st.error("An unexpected error occurred.")
        else:
            st.error("Please enter a valid URL.")

    # Display URL summaries with improved formatting
    if st.session_state.all_summaries:
        st.subheader("URL Summaries")
        for url, summaries in st.session_state.all_summaries.items():
            with st.expander(f"Summary for {url}", expanded=True):
                if summaries["extractive"].strip() == "Extractive summary not found.":
                    st.warning("Could not generate extractive summary")
                else:
                    st.subheader("Extractive Summary")
                    st.markdown(summaries["extractive"])
                
                st.subheader("Abstractive Summary")
                st.markdown(summaries["abstractive"])
                
                st.subheader("Highlights & Analysis")
                st.markdown(summaries["highlights"])
                
                st.write("---")

    # Excel file processing
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        if st.session_state.prev_upload != uploaded_file.name:
            st.session_state.processed_df = pd.read_excel(uploaded_file)
            st.session_state.prev_upload = uploaded_file.name
            st.session_state.processing_complete = False
            st.session_state.all_summaries = {}
            
            for col in ['Model', 'Extractive Summary', 'Abstractive Summary', 'Highlights & Analysis']:
                if col not in st.session_state.processed_df.columns:
                    st.session_state.processed_df[col] = ''

        df = st.session_state.processed_df
        
        if 'BillState' not in df.columns or 'BillTextURL' not in df.columns:
            st.error("File must contain 'BillState' and 'BillTextURL' columns")
        else:
            if st.button("Process Excel File"):
                with st.spinner('Processing URLs... This may take several minutes'):
                    total_urls = len(df['BillTextURL'].dropna())
                    processed_count = 0
                    
                    for index, row in df.iterrows():
                        url = row['BillTextURL']
                        if pd.notna(url) and df.at[index, 'Model'] == '':
                            processed_count += 1
                            try:
                                result = process_input(url)
                                status_msg = f"Processing URL {processed_count}/{total_urls}: {url}"
                                
                                if isinstance(result, dict) and "error" in result:
                                    st.warning(f"{status_msg} - Can't generate summary")
                                    df.at[index, 'Model'] = result.get('model', 'gpt-4')
                                    df.at[index, 'Extractive Summary'] = "Error"
                                    df.at[index, 'Abstractive Summary'] = "Error"
                                    df.at[index, 'Highlights & Analysis'] = "Error"
                                else:
                                    st.success(f"{status_msg} - Completed")
                                    
                                    df.at[index, 'Model'] = result.get('model', 'gpt-4')
                                    df.at[index, 'Extractive Summary'] = result.get('extractive', 'Error')
                                    df.at[index, 'Abstractive Summary'] = result.get('abstractive', 'Error')
                                    highlights = format_highlights(result.get('highlights', 'Error'))
                                    df.at[index, 'Highlights & Analysis'] = highlights

                            except Exception as e:
                                logging.error(f"Error processing {url}: {str(e)}")
                                st.warning(f"{status_msg} - Can't generate summary")
                                df.at[index, 'Model'] = "Error"
                                df.at[index, 'Extractive Summary'] = "Error"
                                df.at[index, 'Abstractive Summary'] = "Error"
                                df.at[index, 'Highlights & Analysis'] = "Error"
                                continue

                    st.session_state.processed_df = df
                    st.session_state.processing_complete = True

                st.success("Processing complete! You can now download the file.")

            # Display summaries from uploaded Excel file
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                st.subheader("Summaries from Excel File")
                df = st.session_state.processed_df
                
                for index, row in df.iterrows():
                    url = row.get("BillTextURL", "")
                    extractive_summary = row.get("Extractive Summary", "No summary available")
                    abstractive_summary = row.get("Abstractive Summary", "No summary available")
                    highlights = row.get("Highlights & Analysis", "No highlights available")

                    if pd.notna(url) and url.strip():
                        with st.expander(f"Summary for {url}", expanded=False):
                            if extractive_summary.strip() == "Error":
                                st.warning("Could not generate extractive summary")
                            else:
                                st.subheader("Extractive Summary")
                                st.markdown(extractive_summary)
                            
                            st.subheader("Abstractive Summary")
                            st.markdown(abstractive_summary)
                            
                            st.subheader("Highlights & Analysis")
                            st.markdown(highlights)
                            
                            st.write("---")

            # Download button
            # Ensure the file name is derived from the uploaded file
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.processed_df.to_excel(writer, index=False)

                # Extract the last 7 characters from the uploaded file name (excluding the extension)
                original_filename = st.session_state.prev_upload
                base_name = original_filename.rsplit('.', 1)[0]  # Remove extension
                last_7_chars = base_name[-7:] if len(base_name) >= 7 else base_name

                # Format the final filename
                processed_filename = f"{last_7_chars}_summarized.xlsx"

                # Provide download button with the new name
                st.download_button(
                    label="Download Updated Excel",
                    data=output.getvalue(),
                    file_name=processed_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download_btn'
                )


if __name__ == "__main__":
    main()
