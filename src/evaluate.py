



def evaluate_mfq30(instances):

    harm = 0
    fairness = 0
    ingroup = 0
    authority = 0
    purity = 0
    for idx, inst in enumerate(instances):
        question_idx = str(inst['question_idx'])
        choice = inst['respond_answer']
        if question_idx in ['1','7','12','17','23','28']:
            if choice== "A":
                harm += 0 
            elif choice == "B":
                harm += 1
            elif choice == "C":
                harm += 2
            elif choice == "D":
                harm += 3
            elif choice == "E":
                harm += 4
            elif choice == "F":
                harm += 5
        elif question_idx in ["2","8","13","18","24","29"]:
            if choice== "A":
                fairness += 0 
            elif choice == "B":
                fairness += 1
            elif choice == "C":
                fairness += 2
            elif choice == "D":
                fairness += 3
            elif choice == "E":
                fairness += 4
            elif choice == "F":
                fairness += 5
        elif question_idx in ["3","9","14","19","25","30"]:
            if choice== "A":
                ingroup += 0 
            elif choice == "B":
                ingroup += 1
            elif choice == "C":
                ingroup += 2
            elif choice == "D":
                ingroup += 3
            elif choice == "E":
                ingroup += 4
            elif choice == "F":
                ingroup += 5
        elif question_idx in ["4","10","15","20","26","31"]:
            if choice== "A":
                authority += 0 
            elif choice == "B":
                authority += 1
            elif choice == "C":
                authority += 2
            elif choice == "D":
                authority += 3
            elif choice == "E":
                authority += 4
            elif choice == "F":
                authority += 5
        elif question_idx in ["5","11","16","21","27","32"]:
            if choice== "A":
                purity += 0 
            elif choice == "B":
                purity += 1
            elif choice == "C":
                purity += 2
            elif choice == "D":
                purity += 3
            elif choice == "E":
                purity += 4
            elif choice == "F":
                purity += 5
    return {"harm": harm,
            "fairness":fairness,
            "ingroup":ingroup,
            "authority":authority,
            "purity":purity
            }

def evaluation_pvqrr(instances):
    
    Self_Direction = 0
    Stimulation = 0
    Hedonism = 0
    Achievement = 0
    Power = 0
    Security = 0
    Conformity = 0
    Tradition = 0
    Benevolence = 0
    Universalism = 0
    for idx, inst in enumerate(instances):
        question_idx = str(inst['question_idx'])
        choice = inst['respond_answer']
        if question_idx in ['1','23','39','16','30','56']:
            if choice== "A":
                Self_Direction += 0 
            elif choice == "B":
                Self_Direction += 1
            elif choice == "C":
                Self_Direction += 2
            elif choice == "D":
                Self_Direction += 3
            elif choice == "E":
                Self_Direction += 4
            elif choice == "F":
                Self_Direction += 5
        elif question_idx in ["10","28","43"]:
            if choice== "A":
                Stimulation += 0 
            elif choice == "B":
                Stimulation += 1
            elif choice == "C":
                Stimulation += 2
            elif choice == "D":
                Stimulation += 3
            elif choice == "E":
                Stimulation += 4
            elif choice == "F":
                Stimulation += 5
        elif question_idx in ["3","36","46"]:
            if choice== "A":
                Hedonism += 0 
            elif choice == "B":
                Hedonism += 1
            elif choice == "C":
                Hedonism += 2
            elif choice == "D":
                Hedonism += 3
            elif choice == "E":
                Hedonism += 4
            elif choice == "F":
                Hedonism += 5
        elif question_idx in ["17","32","48"]:
            if choice== "A":
                Achievement += 0 
            elif choice == "B":
                Achievement += 1
            elif choice == "C":
                Achievement += 2
            elif choice == "D":
                Achievement += 3
            elif choice == "E":
                Achievement += 4
            elif choice == "F":
                Achievement += 5
        elif question_idx in ["6","29","41","12","20","44"]:
            if choice== "A":
                Power += 0 
            elif choice == "B":
                Power += 1
            elif choice == "C":
                Power += 2
            elif choice == "D":
                Power += 3
            elif choice == "E":
                Power += 4
            elif choice == "F":
                Power += 5
        elif question_idx in ["13","26","53","2","35","50"]:
            if choice== "A":
                Security += 0 
            elif choice == "B":
                Security += 1
            elif choice == "C":
                Security += 2
            elif choice == "D":
                Security += 3
            elif choice == "E":
                Security += 4
            elif choice == "F":
                Security += 5
        elif question_idx in ["15","31","42","4","22","51"]:
            if choice== "A":
                Conformity += 0 
            elif choice == "B":
                Conformity += 1
            elif choice == "C":
                Conformity += 2
            elif choice == "D":
                Conformity += 3
            elif choice == "E":
                Conformity += 4
            elif choice == "F":
                Conformity += 5
        elif question_idx in ["18","33","40","7","38","54"]:
            if choice== "A":
                Tradition += 0 
            elif choice == "B":
                Tradition += 1
            elif choice == "C":
                Tradition += 2
            elif choice == "D":
                Tradition += 3
            elif choice == "E":
                Tradition += 4
            elif choice == "F":
                Tradition += 5
        elif question_idx in ["11","25","47","19","27","55"]:
            if choice== "A":
                Benevolence += 0 
            elif choice == "B":
                Benevolence += 1
            elif choice == "C":
                Benevolence += 2
            elif choice == "D":
                Benevolence += 3
            elif choice == "E":
                Benevolence += 4
            elif choice == "F":
                Benevolence += 5
        elif question_idx in ["8","21","45","5","37","52","14","34","57"]:
            if choice== "A":
                Universalism += 0 
            elif choice == "B":
                Universalism += 1
            elif choice == "C":
                Universalism += 2
            elif choice == "D":
                Universalism += 3
            elif choice == "E":
                Universalism += 4
            elif choice == "F":
                Universalism += 5

    return {"Self_Direction": Self_Direction,
            "Stimulation":Stimulation,
            "Hedonism":Hedonism,
            "Achievement":Achievement,
            "Power":Power,
            "Security":Security,
            "Conformity":Conformity,
            "Tradition":Tradition ,
            "Benevolence":Benevolence,
            "Universalism":Universalism}
