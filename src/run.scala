
import breeze.linalg.DenseVector
import org.apache.spark._
import org.apache.log4j._




class Parser extends java.io.Serializable {
  def parseFeatures(line:String):DenseVector[Double] = {
    val fields = DenseVector(line.split(",").dropRight(1).map(x=>x.toDouble))
    val v = DenseVector.ones[Double](fields.length + 1)
    v(1 to fields.length ) := fields
    v
  }

  def parseLabel(line:String):Double = {
    val fields = line.split(",")
    if (fields.takeRight(1)(0) == "R") 1.0 else 0.0
  }
}


object run {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "RatingsCounter")

    val parser = new Parser()
    val lines = sc.textFile("./data/sonar.all-data.csv")
    val Array(train_data,test_data) = lines.randomSplit(Array(0.90, 0.10))



    val train_features = train_data.map(parser.parseFeatures)
    val train_labels = train_data.map(parser.parseLabel)
    val test_features = test_data.map(parser.parseFeatures)
    val test_labels = test_data.map(parser.parseLabel)




    val model : Perceptron = new Perceptron(0.01f,500, "threshold")
    model.fit(train_features, train_labels)


    val X_y = test_features.zip(test_labels)
    val accuracy = X_y.map(data => {
      if (Math.abs(model.prediction(data._1) - data._2) <0.01) 1.0 else 0.0
    }).reduce((x,y) => x + y) / test_features.count() * 100

    println(f"Accuracy on test data:  $accuracy%2.2f%%")



  }

}