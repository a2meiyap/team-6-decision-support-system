# Team 6's Decision Support System

A web app that aims to improve students' study habits using Linear Regression!

## Repository Contents

* `msci_436_group_6_UI.py`: Script to run Streamlit web app
* `MSCI436 Project Milestone 2.pdf`: Report synthesizing data, model, user interface, and insights 
* `MSCI 436 Student Grade ML Model - Legend.pdf`: Legend for users to refer to when using tool
* `student-por.csv`: Training dataset for ML model

## Application Setup

1. Download all files and save them in a common working directory
2. Open the Terminal in any IDE of your choice (e.g. VS Code, PyCharm)
3. Run the command `streamlit run msci_436_group_6_UI.py` to launch the web app in your local host

<img width="698" alt="image" src="https://user-images.githubusercontent.com/77274093/179047337-fa94fe6b-a2ff-49a8-8d5e-d5b467a00d7b.png">

## Usage Manual
1. Open the left sidebar and toggle the scrollbars according to your or your student's information
     
     * Refer to the legend via the [URL in the sidebar](shorturl.at/cfUY7) or the `MSCI 436 Student Grade ML Model - Legend.pdf` file when making your selections

2. If there are any parameters that you don't want to take into consideration (e.g. Age), select the variable(s) under the Additional Preprocessing Section
3. View the user input parameters that you've selected in the generated table and confirm that they are correct
4. Configure training parameters (i.e. validation dataset size, random seed value) to alter model bias and accuracy
5. Understand which factors impact your grades (both negatively and positively) via the Feature Importance graph and make use of that information in your upcoming tests
6. Receive your expected grade! 
7. Play around with the side bar parameters to figure out what you can do to improve your score

<br>

![image](https://user-images.githubusercontent.com/77274093/179047129-948480e4-db82-4dfb-bb89-3bf975a5cb5b.png)

