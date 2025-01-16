python -m venv venv

call venv\Scripts\activate    

python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt

call streamlit run main.py

pause
