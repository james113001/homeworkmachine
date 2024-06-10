
class FileLoader {
    INPUT_LOCATION = ""

    loadFile():string{
        return "First Question"
    }
}


class Autogen {
    init(){
        const BERT_CONFIGURATION = {}
        const DEEPSEEK_CONFIGURATION = {}

        // Runs some steps and is ready to use by end of it
    }

    makeAttempt(question:string){
            console.log("The answer to " + question + " is...")
            return 0
    }
}

class ProblemSolver {
    predict(model:Autogen,question): number {
        return -1
    }

    predictSome(arrOfQuestions: []) {

    }

    savePrediction(prediction:number) {

    }

    loadQuestionFromFile() {

    }


}

const ag = new Autogen()
const solver = new ProblemSolver()
const question = solver.loadQuestionFromFile()
const prediction = solver.predict(ag,question)
console.log(prediction)
solver.savePrediction(prediction)

