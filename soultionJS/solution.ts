class Bert {
    predictProblemType(:string):{

    }
}
class DeepSeek{
    makeAttempt(){

    }
}
class Autogen {

    const bert = new Bert()
    const ds = new DeepSeek()
    // Make a prediction for problem type
    // make an attempt at solving the problem
    init(){

        const BERT_CONFIGURATION = {}
        const DEEPSEEK_CONFIGURATION = {}

        // Runs some steps and is ready to use by end of it
    }

    predict(question:string): number {
        console.log("The answer to " + question + " is...")
        return 0    }

    predictSome(arrOfQuestions: []) {

    }
}
class ProblemSolver {
    savePrediction(prediction:number) {

    }


    loadQuestionFromFile():string {
        return "What is 1+1"
    }


}

const ag = new Autogen()
const bert = new Bert()
const solver = new ProblemSolver()
const question = solver.loadQuestionFromFile()
const problemType = bert.predictProblemType(question)
// const attemptAtSolving = ds.makeAttempt(problemType,question)
// const prediction = ag.predict(question)
const answer = ag.predict(question);
console.log(answer)
solver.savePrediction(answer)

