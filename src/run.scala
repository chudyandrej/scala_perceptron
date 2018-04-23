
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
    val features = lines.map(parser.parseFeatures)
    val labels = lines.map(parser.parseLabel)



    val a : Perceptron = new Perceptron(0.01f,2000, "sigmoid")
    a.fit(features, labels)


  }

}