import breeze.linalg.DenseVector
import org.apache.spark._
import org.apache.log4j._
import org.apache.spark.sql.{Row, SparkSession}





class Parser extends java.io.Serializable {

  def parseFeatures(line:String):DenseVector[Double] = {
    val features = DenseVector(line.split(",").dropRight(1).map(x=>x.toDouble))
    val v = DenseVector.ones[Double](features.length + 1)
    v(1 to features.length ) := features
    v
  }

  def parseLabel(line:String):Double = {
    val features = line.split(",")
//    if (features.takeRight(1)(0) == "Iris-setosa") 0.0
//    else if (features.takeRight(1)(0) == "Iris-versicolor") 1.0
//    else 2.0

//    if (features.takeRight(1)(0) == "R") 1.0 else 0.0
    features.takeRight(1)(0).toDouble
  }
}


object run {
//TODO unsuccessful try to parse, problem is Any data type in Row object
//  def parse_data(features: Array[Row]): Unit = {
//    println("Start")
//
//    val featuresDV = features.map(r =>{
//      val f_DV = DenseVector(r.toSeq.toArray.foreach(x=>x.to))
//      val v = DenseVector.ones[Double](f_DV.length )
//      println(f_DV.length)
//      v(1 to f_DV.length):=f_DV
//      v
//    })
//    featuresDV.foreach(println)
//    featuresDV(featuresDV.count().toInt)
//
//
//  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "Perceptron")


    //TODO I don't know how convert Dataset type in "data_csv" to vector :(
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("my-spark-app")
//      .config("spark.some.config.option", "config-value")
      .getOrCreate()

    val data_csv = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .load("./data/sonar.all-data.csv")


    val parser = new Parser()
    val lines = sc.textFile("./data/multiclass_data.csv")

    val Array(train_data,test_data) = lines.randomSplit(Array(0.77, 0.33))


    val train_features = train_data.map(parser.parseFeatures)
    val train_labels = train_data.map(parser.parseLabel)
    val test_features = test_data.map(parser.parseFeatures)
    val test_labels = test_data.map(parser.parseLabel)



    val act_f = new PerceptronActFunction

    val model: MultiClassPerceptron = new MultiClassPerceptron(0.005f,500, act_f.sigmoid)
    model.fit(train_features, train_labels)

//    val model : Perceptron = new Perceptron(0.01f,500, act_f.threshold)
//    model.fit(train_features, train_labels)
//
//
    val X_y = test_features.zip(test_labels)
    val accuracy = X_y.map(data => {
      if (Math.abs(model.prediction(data._1) - data._2) <0.01) 1.0 else 0.0
    }).reduce((x,y) => x + y) / test_features.count() * 100

    println(f"Accuracy on test data:  $accuracy%2.2f%%")



  }

}
