import breeze.linalg.DenseVector
import breeze.numerics.{exp, pow}
import org.apache.spark.rdd.RDD




class Perceptron(var learning_rate: Double, var n_epoch: Int, var act_function:String) extends java.io.Serializable  {

  var w: DenseVector[Double] = DenseVector.zeros[Double](1)
  val act_f:String = act_function

  def fit(X: RDD[DenseVector[Double]], y: RDD[Double]): Unit = {

    this.w = DenseVector.zeros[Double](X.first().length)
    val X_y = X.zip(y).cache()

    for(e <- 1 to n_epoch) {

      val delta_w = X_y.map(data => {
        val pred = prediction(data._1)
        learning_rate * (data._2 - pred) * data._1

      })
      this.w += delta_w.reduce((x, y) => x + y)

      if (e % 10 == 0 || e == n_epoch){
        val accuracy = X_y.map(data => {
          if (Math.abs(prediction(data._1) - data._2) <0.01) 1.0 else 0.0
        }).reduce((x,y) => x + y) / X.count()
        println(f"[Epoch $e%d] Accuracy  -----> $accuracy%2.2f")

      }
    }
  }




  def prediction(features: DenseVector[Double]): Double = {

    val act_f = this.act_function match {
      case "sigmoid"  => sigmoid
      case "gaussian"  =>  gaussian
      case "threshold"  =>  threshold
      case _  => threshold
    }

    act_f(features dot this.w)
  }

  val threshold: (Double) => Double = (x) => {
    if (x > 0) 1.0f else 0.0f
  }

  val sigmoid: (Double) => Double = (x) => {
    if (1 / (1 + exp(-x)) > 0.5) 1.0f else 0.0f
  }

  val gaussian: (Double) => Double = (x) => {
    if (exp(- pow(-x,2) / (2*2)) > 0.5) 1.0f else 0.0f
  }

}



