class InputHelper():
    def __init__(self) -> None:
        remote = self.yesno('Is this job remote? y/n')
        has_logo = self.yesno("Does this company have a logo? y/n")
        has_questions = self.yesno("Are there company questions? y/n")

    def yesno(self,question):
        yes = {'yes', 'y'}
        no = {'no', 'n'}  # pylint: disable=invalid-name

        done = False
        print(question)
        while not done:
            choice = input().lower()
            if choice in yes:
                return True
            elif choice in no:
                return False
            else:
                print("Please respond by yes or no.") 

    