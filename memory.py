# There are 3 types of memory
# 1. Conversational Memory: Keeps track of what has been in the context
# 2. Retrieval Memory : Reuse retrieved context 
# 3. Persistent Memory: If you ask after two days. 


class ConversationalMemory:
    def __init__(self):
        self.messages = []
    
    def add_user(self, text):
        self.messages.append(
            {
                "role": "user",
                "content": text
            }
        )
    
    def add_assistant(self, text):
        self.messages.append(
            {
                "role": "assistant",
                "content": text
            }
        )
    
    def get_history(self):
        history = ""

        for message in self.messages:
            history += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )
        return history