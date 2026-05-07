from pyspark.sql.functions import col as spark_col
from spark_ai.udf.sentiment_udf import sentiment_udf

class AI:
    """Public API for AI-powered DataFrame transformations."""
    
    def sentiment(self, column_name: str):
        """Apply sentiment analysis on a column.
        
        Args:
            column_name: Name of the text column.
            
        Returns:
            pyspark.sql.Column with POSITIVE/NEGATIVE labels.
        """
        return sentiment_udf(spark_col(column_name))
