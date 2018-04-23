import breeze.linalg.DenseVector
import breeze.numerics.{exp, pow}
import org.apache.spark.rdd.RDD




class Perceptron(var learning_rate: Double, var n_iter: Int, var act_f:String) extends java.io.Serializable  {

  def fit(X: RDD[DenseVector[Double]], y: RDD[Double]): Unit = {

    var w = DenseVector.zeros[Double](X.first().length)
    val X_y = X.zip(y).cache()

    for(_ <- 1 to n_iter) {

      val delta_w = X_y.map(data => {
        val pred = prediction(data._1, w, act_f)
        learning_rate * (data._2 - pred) * data._1

      })
      w += delta_w.reduce((x, y) => x + y)
    }

    val accuracy = X_y.map(data => {
      if (Math.abs(prediction(data._1, w,act_f) - data._2) <0.01) 1.0 else 0.0
    }).reduce((x,y) => x + y) / X.count()
    println(accuracy)

  }


  def prediction(features: DenseVector[Double], w:DenseVector[Double], activation: String): Double = {
    val tmp = features dot w
    var act_f = threshold
    if(activation == "sigmoid"){
      act_f = sigmoid
    }else if(activation == "gaussian"){
      act_f = gaussian
    }
    act_f(tmp)
  }

  val threshold: (Double) => Double = (x) => {
    println(x)

    if (x > 0) 1.0f else 0.0f
  }

  val sigmoid: (Double) => Double = (x) => {
    println(x)
    if (1 / (1 + exp(-x)) > 0.5) 1.0f else 0.0f
  }

  val gaussian: (Double) => Double = (x) => {
    if (exp(- pow(-x,2) / (2*2)) > 0.5) 1.0f else 0.0f
  }

}



