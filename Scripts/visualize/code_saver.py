def save_LLM2_output(code,Question):
    file_path = Question.replace(" ","_")+".py"
    with open(file_path,"w",encoding="utf-8") as f:
        f.write(code)
        return file_path
