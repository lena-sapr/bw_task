
import pandas as pd
import numpy as np
import spacy
# !python -m spacy download ru_core_news_sm
import ru_core_news_sm

data = pd.read_csv('test_data.csv')

data['text'] = data['text'].apply(lambda x: x.lower())

nlp = spacy.load('ru_core_news_sm')

# Загрузим предобученную модель
nlp_1 = nlp.from_disk('./model_spacy')

greetings = ['здравствуйте', 'здраствуйте', 'добрый день', 
             'добрый вечер', 'доброе утро', 'доброго времени суток']

farewells = ['до свидания', 'до свиданья', 'до связи', 'до созвона', 'до вечера', 'до завтра', 'до встречи',
             'до скорого', 'до новых встреч',
             'будьте здоровы', 'хорошего дня', 'хорошего вечера', 
             'удачи', 'успехов', 'счастливо', 
             'всего доброго', 'всего хорошего', 'всего наилучшего', 
             'прощайте', 'позвольте попрощаться', 'разрешите попрощаться']


def is_greeting(text):
    flag = 0
    for greeting in greetings:
        if text.find(greeting) >= 0:
            flag = 1
            break
            
    return int(flag)


def is_farewell(text):
    flag = 0
    for farewell in farewells:
        if text.find(farewell) >= 0:
            flag = 1
            break
            
    return int(flag)


# Извлекать реплики с приветствием – где менеджер поздоровался. Создаем отдельную колонку (0 - не поздоровался, 1 - поздоровался) 
data['mng_greeted'] = data[data['role'] == 'manager']['text'].apply(is_greeting)
# Извлекать реплики, где менеджер попрощался. Создаем отдельную колонку (0 - не попрощался, 1 - попрощался) 
data['mng_farewell'] = data[data['role'] == 'manager']['text'].apply(is_farewell)


data['mng_greeted'] = data['mng_greeted'].fillna(0)
data['mng_farewell'] = data['mng_farewell'].fillna(0)

# NER для извлечения имени (PER)
def extract_person(text):
    doc = nlp(text)
    
    for ent in doc.ents:
        if (ent.label_ == 'PER'):
            return ent.text
        
    return ''        


# Извлекать реплики, где менеджер представил себя. 
# Извлекать имя менеджера. 
def manager_introduced(text):
    intros = ['меня зовут', 'зовут', 'это', 'разрешите представиться']
    flag = 0 
    for intro in intros:
        if text.find(intro) >= 0:
            flag = 1
   
    name = extract_person(text)
    
    if flag and (name != ''):
        return 1
    else:
        return 0
            
def manager_name(text):
    intros = ['меня зовут', 'зовут', 'это', 'разрешите представиться']
    flag = 0 
    for intro in intros:
        if text.find(intro) >= 0:
            flag = 1

    name = extract_person(text)
    
    if flag and (name != ''):
        return name
    else:
        return ''
            
# Создаем отдельную колонку с именем менеджера, если менеджер представился. Если не представился, то значение пусто 
data['mng_introduced'] = data[data['role'] == 'manager']['text'].apply(manager_introduced)
data['mng_name'] = data[data['role'] == 'manager']['text'].apply(manager_name)

data['mng_introduced'] = data['mng_introduced'].fillna(0)
data['mng_name'] = data['mng_name'].fillna('')

# NER для извлечения компании (ORG)
def extract_org(df):
    if df['role'] == 'manager':
        doc = nlp(df['text'])
    else:
        return ''
    
    for ent in doc.ents:
        if (ent.label_ == 'ORG') and (df['line_n'] < 5):
            return ent.text
        
    return ''  

# Создаем отдельную в колонку с названием компании менеджера, если менеджер ее назвал. Если не назвал, то значение пусто 
data['company_intro'] = data.apply(extract_org, axis=1)
data['company_intro'] = data['company_intro'].fillna('')

# Группируем информацию по диалогам, делаем табличку с аггрегированной информацией по каждому диалогу 
# - поздоровался, 
# - попрощался, 
# - и поздоровался, и попрощался
# - имя менеджера, если представился
# - компания менеджера

def my_aggregate(dfgroup):
    return pd.Series({
        'mng_greeted': any(list(dfgroup['mng_greeted'])),
        'mng_farewelled': any(list(dfgroup['mng_farewell'])),
        'mng_greet_and_bye': any(list(dfgroup['mng_greeted'])) and any(list(dfgroup['mng_farewell'])),
        'mng_introduced': any(list(dfgroup['mng_introduced'])),
        'mng_name': dfgroup[dfgroup['mng_introduced']==1]['mng_name'].values[0] if any(list(dfgroup['mng_introduced'])) else '',
        'mng_company_introduced': ''.join(list(dfgroup['company_intro']))
        
    })

summary = data.groupby('dlg_id').apply(my_aggregate)


print('Summary per dialog:')
print(summary)

summary.to_csv('summary_dialogs.csv')

summary_dict = dict(summary['mng_greet_and_bye'])

# Проверять требование к менеджеру: «В каждом диалоге обязательно необходимо поздороваться и попрощаться с клиентом»
data['mng_greet_and_farewell'] = data['dlg_id'].map(summary_dict)

data.to_csv('test_data_processed.csv')

