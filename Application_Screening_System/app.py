
from flask import Flask,render_template,url_for,request,jsonify
from sklearn.externals import joblib

app = Flask(__name__)
# load the model
import string
import pandas as pd
import datetime
import flask
import numpy as np


#-------- GET USER INPUT HERE -----------#
@app.route('/')
def home():
	return render_template('home.html')

def text_feature_extraction(cvec_pkl, text_array):
    text_df = pd.DataFrame(cvec_pkl.transform(text_array).toarray(), columns=cvec_pkl.get_feature_names())
    return text_df

str_to_remove = string.punctuation
str_to_remove = str_to_remove.replace("%","").replace("-","")

def cleaning_text(string):
	string = string.replace('\\r','').replace('\\n','').replace('\r','').replace('\n','')
	clean_string = "".join([char for char in string if char not in str_to_remove])
	return clean_string


@app.route('/submit',methods=['POST'])
def submit():
	if request.method == 'POST':

		project_no = request.form['project_no']
		teacher_id = request.form ['teacher_id']
		teacher_prefix = request.form['teacher_prefix']
		teacher_number_of_previously_posted_projects = request.form['teacher_number_of_previously_posted_projects']
		school_state = request.form['school_state']
		project_grade_category = request.form['project_grade_category']
		project_subject_categories = request.form['project_subject_categories']
		project_subject_subcategories = request.form['project_subject_subcategories']
		item_quantity = request.form['item_quantity']
		item_price = request.form['item_price']
		project_title = request.form['project_title']
		project_resource_summary = request.form['project_resource_summary']
		project_essay_1_2 = request.form['project_essay_1_2']
		project_essay_3_4 = request.form['project_essay_3_4']
		todays_date = datetime.datetime.now().date()

		feature_df = open('feature_columns.pkl','rb')
		feature_col_df = joblib.load(feature_df)
		test_purpose = feature_col_df.copy()

		# Compute total budget and add to feature
		feature_col_df['budget'] = int(item_quantity) * float(item_price)

		school_state_col = 'sch_state_' + school_state
		#print(school_state_col)
		if {school_state_col}.issubset(feature_col_df.columns):
			feature_col_df[school_state_col] = 1

		month_col = 'month_' + str(todays_date.month)
		#print(month_col)
		if {month_col}.issubset(feature_col_df.columns):
			feature_col_df[month_col] = 1

		day_col = 'day_' + str(todays_date.day)
		#print(day_col)
		if {day_col}.issubset(feature_col_df.columns):
			feature_col_df[day_col] = 1

		subcat_col = 'subcat_' + project_subject_subcategories
		#print(subcat_col)
		if {subcat_col}.issubset(feature_col_df.columns):
			feature_col_df[subcat_col] = 1

		cvec_title = open('cvec_project_title.pkl','rb')
		cvec_p_title = joblib.load(cvec_title)
		project_title = [project_title]
		project_title[0] = cleaning_text(project_title[0])
		project_title_features = text_feature_extraction(cvec_p_title,project_title)
		project_title_features.columns = ['proj_title_'+ x for x in project_title_features.columns]
		for i,value in enumerate(project_title_features.values[0]):
			if value == 1:
				feature_col_df[project_title_features.columns[i]] = 1

		cvec_pr_summary = open('cvec_project_resource_summary.pkl','rb')
		cvec_project_resource_summary = joblib.load(cvec_pr_summary)
		project_resource_summary = [project_resource_summary]
		project_resource_summary[0] = cleaning_text(project_resource_summary[0])
		project_resource_summary_features = text_feature_extraction(cvec_project_resource_summary,project_resource_summary)
		project_resource_summary_features.columns = ['proj_resource_sum_'+ x for x in project_resource_summary_features.columns]
		for i,value in enumerate(project_resource_summary_features.values[0]):
			if value == 1:
				feature_col_df[project_resource_summary_features.columns[i]] = 1

		cvec_pe_1_2 = open('cvec_project_essay_1_2.pkl','rb')
		cvec_project_essay_1_2 = joblib.load(cvec_pe_1_2)
		project_essay_1_2 = [project_essay_1_2]
		project_essay_1_2[0] = cleaning_text(project_essay_1_2[0])
		project_essay_1_2_features = text_feature_extraction(cvec_project_essay_1_2,project_essay_1_2)
		project_essay_1_2_features.columns = ['project_essay_1_2_'+ x for x in project_essay_1_2_features.columns]
		for i,value in enumerate(project_essay_1_2_features.values[0]):
			if value == 1:
				feature_col_df[project_essay_1_2_features.columns[i]] = 1

		cvec_pe_3_4 = open('cvec_project_essay_3_4.pkl','rb')
		cvec_project_essay_3_4 = joblib.load(cvec_pe_3_4)
		project_essay_3_4 = [project_essay_3_4]
		project_essay_3_4[0] = cleaning_text(project_essay_3_4[0])
		project_essay_3_4_features = text_feature_extraction(cvec_project_essay_3_4,project_essay_3_4)
		project_essay_3_4_features.columns = ['project_essay_3_4_'+ x for x in project_essay_3_4_features.columns]
		for i,value in enumerate(project_essay_3_4_features.values[0]):
			if value == 1:
				feature_col_df[project_essay_3_4_features.columns[i]] = 1

		#print(feature_col_df.columns.difference(test_purpose.columns))

		application_model = open('application_model.pkl','rb')
		model = joblib.load(application_model)
		predict_application = model.predict(feature_col_df.values)

		return render_template('result.html'
			,project_no = project_no
			,prediction = predict_application)
    

if __name__ == '__main__':
    app.run(debug=True)

