import random

from utils.config_loader import load_config

config = load_config()


# La génération par LLM va etre fait au moment de la création du summary index.
# cela prend trop de temps ici.
# l'objectif est de mutualiser la fonctionnalités conversations starters pour les deux chats
# llm = load_model(streaming=True)
#
# context = summary_query_engine.query("Make a complete summary of knowledge available"
#                                      " on following topics {topics}.")
#
# ### Answer question ###
# cs_system_prompt = """You are a helpful solution architect and software engineer assistant.
#     Your users are asking questions on specific topics.\
#     Suggest exactly 6 questions related to the provided context to help them find the information they need. \
#     Suggest only short questions without compound sentences. \
#     Question must be self-explanatory and topic related.
#     Suggest a variety of questions that cover different aspects of the context. \
#     Use the summary of knowledge to generate the question on topics. \
#     Make sure they are complete questions, and that they are related to the topics.
#     Output one question per line. Do not number the questions. Do not group question by topics.
#     DO NOT make a summary or an introduction of your result. Output ONLY the generated questions.
#     DO NOT output chapter per topic. Avoid blank line.
#     Avoid duplicate question. Generate question in French.
#     Questions: """
#
# #         Examples:
# #         What information needs to be provided during IHM launch?
# #         How is the data transferred to the service call?
# #         What functions are involved in API Management?
# #         What does the Exposure function in API Management entail?
#
# cs_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", cs_system_prompt),
#         ("human", "{topics}"
#                   "{summary}"),
#     ]
# )
# output_parser = StrOutputParser()
# model = load_model(streaming=False)
#
# chain = cs_prompt | model | output_parser
# response = chain.invoke({"topics": topics, "summary": context})

def get_conversation_starters(topics: list[str], count:int = 4):

    response = ""

    response_list = [line for line in response.split("\n") if line.strip() != '']
    if len(response_list) > count:
        response_list = random.sample(response_list, count)

    elif len(response_list) < count:
        diff = count - len(response_list)

        suggested_questions = suggested_questions_examples

        # check if 'API' is in topics
        if 'API' in topics:
            suggested_questions.extend(suggested_questions_examples_api)

        # check if 'IHM' is in topics
        if 'IHM' in topics:
            suggested_questions.extend(suggested_questions_examples_ihm)

        all_questions = list(suggested_questions)

        selected_questions = set(response_list)

        while len(selected_questions) < count:
            question = random.choice(suggested_questions)
            selected_questions.add(question)

        response_list = list(selected_questions)
        # for _ in range(min(count, len(all_questions))):
        #     question = random.choice(all_questions)
        #     all_questions.remove(question)
        #     response_list.append(question)

        #additional_questions = random.sample(suggested_questions, diff)
        #response_list.extend(additional_questions)

    return response_list


suggested_questions_examples = [
    "Comment sécuriser les données sensibles ?",
    "Comment assurer l'efficacité des performances ?",
    "En quoi consiste l'Analyse de risque MESARI ?",
    "A quoi sert le Cross Origin Resource Sharing ?",
    "Quels sont les principes de la Content Security Policy ?",
    "Comment garantir la sécurité des échanges entre applications ?",
    "Quelles sont les bonnes pratiques pour assurer la fiabilité des ressources Web ?",
    "Pourquoi suivre les spécifications associées est-il important ?"
]
    # API
suggested_questions_examples_api = [
    "Quels sont les mécanismes d'authentification API ?",
    "Quelles sont les principales fonctionnalités du portail fournisseur ?",
    "Que comprend la fonction d'exposition dans la gestion des API ?",
    "Quelle est la différence entre SOAP et REST ?",
    "Que signifie l'acronyme API ?",
    "Quels formats de données sont couramment utilisés dans les APIs ?",
    "Comment tester et déboguer une API ?",
    "Quels sont les avantages d'utiliser une API ?",
    "Que signifie REST et quels en sont les principes clés ?",
    "Comment gérer les versions dans une API ?",
    "Quels outils permettent de documenter une API ?",
    "Comment implémenter une pagination dans une API ?",
    "Qu'est-ce qu'une architecture d'API ?",
]
suggested_questions_examples_ihm = [
    # IHM
    "Quelles informations doivent être fournies lors du lancement de l'IHM?",
    "Quels sont les principes de base d'une bonne conception d'interface utilisateur ?",
    "Comment rendre une interface utilisateur accessible aux personnes handicapées ?",
    "Quels sont les différents types de composants d'interface utilisateur ?",
    "Comment concevoir une expérience utilisateur cohérente sur différents appareils ?",
    "Quels sont les avantages du design 'mobile first' ?",
    "Comment effectuer des tests d'utilisabilité pour une interface ?",
    "Que signifie 'responsive design' pour une interface web ?",
    "Quels frameworks facilitent le développement d'interfaces utilisateur modernes ?",
    "Comment optimiser les performances d'une interface utilisateur ?",
    "Quelle est l'importance des conventions de conception dans une interface ?",
]
    # AUTRES QUESTIONS API
    # "Comment structurer une API RESTful ?",
    # "Quels sont les bons usages des méthodes HTTP ?",
    # "Comment définir des URIs pour les ressources ?",
    # "Qu'est-ce que HATEOAS et comment l'implémenter ?",
    # "Comment paginer et filtrer des collections de ressources ?",
    # "Quels mécanismes utiliser pour l'authentification API ?",
    # "Comment gérer les versions d'une API ?",
    # "Quelle est la stratégie de contrôle d'accès recommandée ?",
    # "Comment documenter une API efficacement ?",
    # "Comment implémenter le throttling pour une API ?",
    # "Quels sont les principes de conception d'une IHM intuitive ?",
    # "Comment assurer la résilience d'une API ?",
    # "Quels sont les formats standards pour les données API ?",
    # "Comment surveiller la performance d'une API ?",
    # "Quels sont les aspects de sécurité à considérer pour une API ?",
    # "Comment gérer la rétrocompatibilité des API ?",
    # "Quels sont les avantages du caching pour une API ?",
    # "Comment assurer la haute disponibilité d'une API ?",
    # "Quels outils utiliser pour le monitoring d'une API ?",
    # "Comment prévenir les injections dans une API ?"


