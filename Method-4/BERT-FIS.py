import jax
import jax.numpy as jnp
from transformers import BertTokenizer, FlaxBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = FlaxBertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """
    Step 1: BERT Encoding 
    Converts raw text into a 768-dimensional vector.
    """
    inputs = tokenizer(text, return_tensors="jax", padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    
    # Extract the [CLS] token (first token) representation
    # e_BERT = BERT(S) 
    e_bert = outputs.last_hidden_state[:, 0, :] 
    return jnp.squeeze(e_bert)
def type2_fuzzification(x,centers,sigma_upper,sigma_lower):

    # Upper Membership func
    mu_upper=jnp.exp(-((x[:,None]-centers)**2)/(2*sigma_upper**2))

    #Lower Membership func
    mu_lower=jnp.exp(-((x[:,None]-centers)**2)/(2*sigma_lower**2))

    return mu_upper,mu_lower    


class BERT_FIS_Pipeline:
    def __init__(self,num_rules=10,bert_dim=768,num_classes=2):
        self.num_rules=num_rules
        self.bert_dim=bert_dim
        self.num_classes=num_classes

    def init_params(self,key):
        k1,k2=jax.random.split(key,2)
        return {
            'c':jax.random.normal(k1,(self.num_rules,self.bert_dim)),
            'sigma_upper':jnp.ones((self.num_rules,self.bert_dim))*1.2,
            'sigma_lower':jnp.ones((self.num_rules,self.bert_dim))*0.8,
            'theta':jax.random.normal(k2,(self.num_rules,self.bert_dim)),
        }
    

    def forward(self,params,raw_text):

        e_bert=get_bert_embedding(raw_text)


        #Fuzzification
        diff_sq = (e_bert[:, None] - params['c'])**2
        mu_upper = jnp.exp(-diff_sq / (2 * params['sigma_upper']**2))
        mu_lower = jnp.exp(-diff_sq / (2 * params['sigma_lower']**2))

        #Rule Computation (Product T-Norm)
        f_lower=jnp.prod(mu_lower,axis=1)
        f_upper=jnp.prod(mu_upper,axis=1)


        #Type-reduction and Defuzzification
        # y=(sum(f_up* theta)+sum(f_low*theta))/(sum(f_up)+sum(f_low))

        numerator=jnp.dot(f_upper,params['theta'])+jnp.dot(f_lower,params['theta'])
        denominator=jnp.sum(f_upper)+jnp.sum(f_lower)+1e-8

        logits=numerator/denominator

        return logits 