import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return data


def clean_data(data):
    return data.drop_duplicates()


def prepare_data(data):
    X = data["message"]
    y = data["label"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", f"{accuracy:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict_message(model, vectorizer):
    while True:
        message = input("\nEnter a message (or type 'exit' to stop): ")

        if message.lower() == "exit":
            print("Program stopped.")
            break

        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)

        print("Prediction:", prediction[0])


def main():
    data = load_data()

    print("Total messages:", len(data))
    print("\nMessage categories:")
    print(data["label"].value_counts())
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nDuplicate rows:", data.duplicated().sum())

    cleaned_data = clean_data(data)

    print("\nMessages before removing duplicates:", len(data))
    print("Messages after removing duplicates:", len(cleaned_data))
    print("Duplicates removed:", len(data) - len(cleaned_data))
    print("\nMessage categories after removing duplicates:")
    print(cleaned_data["label"].value_counts())

    X_train, X_test, y_train, y_test = prepare_data(cleaned_data)

    print("\nTraining messages:", len(X_train))
    print("Testing messages:", len(X_test))

    model, vectorizer = train_model(X_train, y_train)

    print("\nModel training completed!")

    evaluate_model(model, vectorizer, X_test, y_test)

    predict_message(model, vectorizer)


if __name__ == "__main__":
    main()