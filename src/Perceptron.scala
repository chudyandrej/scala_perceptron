import breeze.linalg.DenseVector
import breeze.numerics.{exp, pow}
import org.apache.spark.rdd.RDD



class MultiClassPerceptron(var learning_rate: Double, var n_epoch: Int, var act_function:(Double, Boolean) => Double) extends java.io.Serializable  {

  var perceptrons:Array[BinaryPerceptron] = new Array[BinaryPerceptron](1)
  var classes = new Array[Double](1)

  def fit(X: RDD[DenseVector[Double]], y: RDD[Double], logging:Boolean = true): Unit = {
    this.classes = y.distinct().collect().array
    val classes_count = this.classes.length
    this.perceptrons = new Array[BinaryPerceptron](classes_count)

    for(i <- 0 until classes_count) {
      this.perceptrons(i) = new BinaryPerceptron(this.learning_rate, this.n_epoch, this.act_function, multiclass=true)
      // Relabeled for class i
      val label_class = y.map(x => {if (x == this.classes(i)) 1.0 else 0.0})

      this.perceptrons(i).fit(X, label_class, logging=false)

    }

//    val X_y = X.zip(y)
//    val accuracy = X_y.map(data => {
//      if (Math.abs(prediction(data._1) - data._2) < 0.01) 1.0 else 0.0
//    }).reduce((x,y) => x + y) / X.count()
//    if(logging) {
//      println(f"Accuracy  -----> $accuracy%2.2f")
//    }
  }

  def prediction(features: DenseVector[Double]): Double = {
    // Multi-Class Decision Rule:
    val scores = this.perceptrons.map(x=>x.prediction(features))
    val max_index = scores.zipWithIndex.maxBy(_._1)._2
    this.classes(max_index)
  }
}




class BinaryPerceptron(var learning_rate: Double, var n_epoch: Int, var act_function:(Double, Boolean) => Double,
                 val multiclass: Boolean = false) extends java.io.Serializable  {

  var w: DenseVector[Double] = DenseVector.ones[Double](1)

  def fit(X: RDD[DenseVector[Double]], y: RDD[Double], logging:Boolean = true): Unit = {

    this.w = DenseVector.zeros[Double](X.first().length)
    val X_y = X.zip(y)

    for(e <- 1 to n_epoch) {

      val delta_w = X_y.map(data => {
        val pred = prediction(data._1)
        learning_rate * (data._2 - pred) * data._1
      })

      this.w += delta_w.reduce((x, y) => x + y)

      if (e % 10 == 0 || e == n_epoch){
        val accuracy = X_y.map(data => {
          if (Math.abs(prediction(data._1) - data._2) < 0.01) 1.0 else 0.0
        }).reduce((x,y) => x + y) / X.count()
        if(logging) {
          println(f"[Epoch $e%d] Accuracy  -----> $accuracy%2.2f")
        }
      }
    }
  }


  def prediction(features: DenseVector[Double]): Double = {
    this.act_function(features dot this.w, this.multiclass)
  }

}

class PerceptronActFunction extends java.io.Serializable {

  val linear: (Double, Boolean) => Double = (x, binary) => {
    if(binary){
      x
    } else {
      if (x > 0) 1.0f else 0.0f
    }

  }

  val sigmoid: (Double, Boolean) => Double = (x, binary) => {
    if(binary) {
      1 / (1 + exp(-x))
    } else {
      if ((1 / (1 + exp(-x))) > 0.5) 1.0f else 0.0f
    }
  }

  val gaussian: (Double, Boolean) => Double = (x, binary) => {
    if(binary) {
      exp(-pow(-x, 2) / (2 * 2))
    }else{
      if (exp(- pow(-x,2) / (2*2)) > 0.5) 1.0f else 0.0f
    }
  }
}



