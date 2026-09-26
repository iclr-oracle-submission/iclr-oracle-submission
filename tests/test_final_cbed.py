"""Behavioral regressions for final-manuscript Appendix G Algorithm 1."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from algorithms.cbed import CascadedBidirectionalDecipherment, DeciphermentResult
from data.time_encoding import get_era_for_time
from data.ccamc_adapter import model_record

class Drift:
    def __init__(self):
        self.forward, self.backward, self.encoded, self.survival = [], [], [], []
    def to(self, d): return self
    def eval(self): return self
    def encode(self, x, t):
        self.encoded.extend(t.tolist())
        return x[:, :2] + t[:, None] * torch.tensor([0., 1.])
    def flow_forward(self, z, a, b):
        self.forward.append((a,b))
        return z + (b-a)*torch.tensor([0.,1.])
    def flow_backward(self, z, a, b):
        self.backward.append((a,b))
        return z + (b-a)*torch.tensor([0.,1.])
    def predict_survival(self, z, t):
        self.survival.extend(t.tolist())
        return torch.full((len(z),1), .8 if float(t[0]) <= .7 else 0.)

def features(z):
    x=torch.zeros(352);x[:2]=torch.tensor(z);return x

def record(c, z, t):
    return dict(character=c,features=features(z),time=t,source='fixture plate')

def database():
    return {e:{e+'A':record('A',[1,0],t),e+'B':record('B',[0,1],t)}
            for e,t in zip(['Bronze','Seal','Clerical','Regular'],[.4,.7,.85,1.])}

def test_progressive_ode_matching_times_partial_paths_and_raw_max_cosine():
    m=Drift();db=database()
    paths={'A':[dict(source='endpoint edition',occurrences={'Regular':'RegularA'}),
                dict(source='partial edition',occurrences={'Regular':'RegularA','Bronze':'BronzeA'})],
           'B':[dict(source='endpoint edition',occurrences={'Regular':'RegularB'})]}
    cb=CascadedBidirectionalDecipherment(m,db,retrieval_depth_K=2,evolution_paths=paths,
                                        pruning_thresholds={'OBI':.1,'Bronze':.1})
    m.encoded.clear();out=cb.decipher_single(features([1,0]),'query',.05)
    assert m.encoded == pytest.approx([.05])  # query encoded once
    assert m.survival == pytest.approx([.35,.7])  # never checked at Regular
    assert out.predicted_character=='A' and out.candidate_scores['A']==pytest.approx(1.)
    assert out.pruned_candidates==['B'] and not out.candidate_log_scores
    assert [c['verification_scope'] for c in out.path_evidence['A']] == ['endpoint_only','available_observations']
    steps=out.path_evidence['A'][1]['checkpoints']
    assert [s['time'] for s in steps]==pytest.approx([.4,.05])
    assert (.4,.35) in m.forward  # recorded Bronze explicitly transported to retrieval time
    assert all(span in m.forward for span in [(.05,.35),(.35,.7),(.7,.85),(.85,1.)])
    assert all(span in m.backward for span in [(1.,.85),(.85,.7),(.7,.35),(.35,.05)])
    assert out.survival_probability==pytest.approx(.8)

def test_accumulated_reach_uses_skip_edges_and_excludes_query_edges():
    m=Drift();db=database()
    # B is not retrieved at Regular, but was reachable from the accumulated Bronze set.
    db['Regular']['RegularB']=record('B',[-1,0],1.)
    cb=CascadedBidirectionalDecipherment(m,db,retrieval_depth_K=1,
        known_correspondences={'BronzeA':{'Regular':['RegularB']}},
        backward_verification=False,survival_check=False)
    out=cb.decipher_single(features([1,0]),'query')
    assert out.forward_candidates['Regular']==['RegularA','RegularB']
    cb.known_correspondences['query']={'Regular':['RegularB']}
    with pytest.raises(ValueError,match='Held-out query'):
        cb.decipher_single(features([1,0]),'query')
    cb.known_correspondences={'BronzeA':{'Seal':['query']}}
    with pytest.raises(ValueError,match='Held-out query'):
        cb.decipher_single(features([1,0]),'query')

def test_rejection_reasons_and_norm_product_floor():
    class Low(Drift):
        def predict_survival(self,z,t): return torch.zeros(len(z),1)
    cb=CascadedBidirectionalDecipherment(Low(),database(),backward_verification=False)
    out=cb.decipher_single(features([1,0]))
    assert out.result_type==DeciphermentResult.ABSTAIN and out.rejection_reason=='low_survival_score'
    cb=CascadedBidirectionalDecipherment(Drift(),database(),evolution_paths={},
                                        pruning_thresholds={'OBI':.1},survival_check=False)
    out=cb.decipher_single(features([1,0]))
    assert out.result_type==DeciphermentResult.ABSTAIN and out.rejection_reason=='no_consistent_candidate'
    assert cb._cosine(torch.tensor([[1e-6,0.]]),torch.tensor([[1e-6,0.]])).item()==pytest.approx(1e-4)

@pytest.mark.parametrize('t,era',[(.299,'OBI'),(.3,'Bronze'),(.699,'Bronze'),(.7,'Seal'),(.85,'Clerical'),(1.,'Regular')])
def test_final_interval_boundaries(t,era):
    assert get_era_for_time(t)==era

def test_scribal_group_not_automatically_chronological():
    row=dict(occurrence_id='obi',character='A',script_type='甲骨文',version_subgroup='賓組',image_url='plate')
    a=model_record(row);row['version_subgroup']='黃組';b=model_record(row)
    assert a['time']==b['time']==.05 and a['scribal_group']!=b['scribal_group']


def test_cycle_population_excludes_complete_chain_records():
    from train import evolution_losses
    from config import MSEFConfig
    class Toy:
        def encode(self,x,t): return x[:, :2]
        def flow_batch(self,z,a,b): return z + a[:,None]
    batch=dict(features_src=torch.zeros(3,352),features_tgt=torch.zeros(3,352),
               time_src=torch.tensor([1.,2.,100.]),time_tgt=torch.tensor([3.,4.,100.]),
               pair_type=['adjacent','skip','complete'])
    _,parts=evolution_losses(Toy(),batch,MSEFConfig(),'cpu')
    assert parts['loss_cyc'].item()==pytest.approx((2*4**2+2*6**2)/2)
    batch['pair_type']=['complete']*3
    _,parts=evolution_losses(Toy(),batch,MSEFConfig(),'cpu')
    assert parts['loss_cyc'].item()==0
