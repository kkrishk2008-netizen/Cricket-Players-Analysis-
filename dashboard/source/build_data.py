import json, numpy as np, pandas as pd
D='/home/claude/proj/SuperRR_Hackathon/data/'; R='/home/claude/runs/full/'
comp=pd.read_csv(D+'competitions.csv'); mt=pd.read_csv(D+'matches.csv').merge(comp[['comp_id','comp_name','match_format','season']],on='comp_id')
pe=pd.read_csv(D+'playing_eleven.csv'); pe=pe[pe.is_played.astype(str).str.lower()=='true']
teams=pd.read_csv(D+'teams.csv').drop_duplicates('team_id').set_index('team_id').team_name
pe=pe.merge(mt[['match_id','match_format','comp_name','season','match_date']],on='match_id'); pe['team']=pe.team_id.map(teams)
t20i=pe[pe.match_format=='T20I']
country=t20i.groupby('player_id').team.agg(lambda s:s.value_counts().index[0])
caps=t20i.groupby('player_id').match_id.nunique()
ipl=pe[pe.comp_name.str.contains('Indian Premier League|IPL',case=False)]
print('IPL comps', ipl.comp_name.unique()[:6])
last_ipl=ipl.sort_values('match_date').groupby('player_id').agg(ipl_team=('team','last'),ipl_season=('season','last'))
ipl_n=ipl.groupby('player_id').match_id.nunique()
b=pd.read_csv(D+'ball_by_ball/ball_by_ball.csv',dtype=str,on_bad_lines='skip',usecols=['match_id','innings','striker_id','batting_order','bowling_type','pace_or_spin','bowler_id','batting_type','overid'])
b=b[b.match_id.str.fullmatch(r'\d+',na=False)]
for c in ['match_id','innings','striker_id','batting_order','bowler_id','overid']: b[c]=pd.to_numeric(b[c],errors='coerce')
bo=b.drop_duplicates(['match_id','innings','striker_id']).groupby('striker_id').batting_order.median()
hand=b.groupby('striker_id').batting_type.agg(lambda s:s.value_counts().index[0] if s.notna().any() else '')
btype=b.groupby('bowler_id').bowling_type.agg(lambda s:s.value_counts().index[0] if s.notna().any() else '')
ps=b.groupby('bowler_id').pace_or_spin.agg(lambda s:s.value_counts().index[0] if s.notna().any() else '')
bl=pd.read_csv(R+'ball_level_spi.csv.gz',usecols=['match_id','innings','over','legal_before','bowler_id','striker_id','wp_bat','LI','spi_bat','spi_bowl','wkts_before','score_before','target','runs_needed'])
bl['phase']=np.select([bl.over<=6,bl.over>=17],['pp','death'],'middle')
ph=bl.groupby(['bowler_id','phase']).size().unstack(fill_value=0); phs=ph.div(ph.sum(1),axis=0)
lg=ph.sum()/ph.sum().sum(); print('league phase share',lg.round(3).to_dict())
def pos(o):
    if pd.isna(o): return 'Unknown'
    return 'Opener' if o<=2 else 'Top order (3)' if o<=3 else 'Middle order (4-5)' if o<=5 else 'Finisher (6-7)' if o<=7 else 'Lower order (8+)'
def common(pid):
    capped = pid in caps.index
    return dict(country=country.get(pid,'') if capped else '', capped=bool(capped), caps=int(caps.get(pid,0)),
                ipl_team=last_ipl.ipl_team.get(pid,''), ipl_season=int(last_ipl.ipl_season.get(pid,0)) if pid in last_ipl.index else None,
                ipl_matches=int(ipl_n.get(pid,0)))
def clean(v):
    if isinstance(v,(float,np.floating)): return None if np.isnan(v) else round(float(v),4)
    if isinstance(v,(np.integer,)): return int(v)
    return v
out={'batters':[],'bowlers':[]}
rb=pd.read_csv(R+'ranking_batters_full.csv')
for _,r in rb.iterrows():
    pid=int(r.player_id); x={k:clean(v) for k,v in r.items()}; x.update(common(pid))
    x['bat_pos']=clean(bo.get(pid,np.nan)); x['position']=pos(bo.get(pid,np.nan)); x['hand']=hand.get(pid,'')
    out['batters'].append(x)
rw=pd.read_csv(R+'ranking_bowlers_full.csv')
for _,r in rw.iterrows():
    pid=int(r.player_id); x={k:clean(v) for k,v in r.items()}; x.update(common(pid))
    s=phs.loc[pid] if pid in phs.index else pd.Series({'pp':0,'middle':0,'death':0})
    x.update(share_pp=clean(s.get('pp',0)),share_middle=clean(s.get('middle',0)),share_death=clean(s.get('death',0)))
    rel=(s/lg).fillna(0); x['primary_phase']={'pp':'Powerplay','middle':'Middle overs','death':'Death'}[rel.idxmax()]
    x['phase_tags']=[n for k,n,th in (('pp','Powerplay',.30),('middle','Middle overs',.45),('death','Death',.25)) if s.get(k,0)>=th]
    x['bowl_type']=btype.get(pid,''); x['pace_spin']=ps.get(pid,'')
    out['bowlers'].append(x)
# pressure explainer data
c2=bl[bl.innings==2]; g=c2.groupby(['wkts_before','over']).LI
hm=g.mean().where(g.size()>=30).unstack().reindex(index=range(10),columns=range(1,21))
out['heatmap']=[[None if pd.isna(v) else round(float(v),3) for v in row] for row in hm.values]
rep=json.load(open(R+'run_report.json'))
out['thresholds']={'bat':rep['spi_thresholds_bat'],'bowl':rep['spi_thresholds_bowl']}
cc=c2.groupby('match_id').agg(li=('LI','mean'),n=('LI','size')); sm=cc[cc.n>=100].li.idxmax()
s=bl[bl.match_id==sm].sort_values(['innings','legal_before']).drop_duplicates(['innings','legal_before'],keep='last')
mrow=mt.set_index('match_id').loc[sm]
out['sample']={'match':f"{mrow.home_team} vs {mrow.away_team}", 'comp':mrow.comp_name,'date':mrow.match_date,'result':mrow.result,
  'balls':[dict(i=int(r.innings),b=int(r.legal_before),wp1=round(float(r.wp_bat if r.innings==1 else 1-r.wp_bat),4),spi=round(float(r.spi_bat),3),li=round(float(r.LI),3),
   sc=int(r.score_before),wk=int(r.wkts_before),need=None if pd.isna(r.runs_needed) else int(r.runs_needed)) for r in s.itertuples()]}
out['meta']={'calib':rep['wp_calibration_oof']['calibrated'],'calib_raw':rep['wp_calibration_oof']['raw_markov'],
  'n_matches':int(bl.match_id.nunique()),'n_balls':int(len(bl)),'exp_total':rep['markov']['expected_first_innings_total'],
  'bat_k':rep['batting_meta']['ks'],'bowl_k':rep['bowling_meta']['ks'],'league_phase_share':{k:round(float(v),3) for k,v in lg.items()}}
json.dump(out,open('/home/claude/dash/data.json','w'),separators=(',',':'))
B=pd.DataFrame(out['batters']); W=pd.DataFrame(out['bowlers'])
print(len(B),len(W),'capped bat',B.capped.mean().round(2),'capped bowl',W.capped.mean().round(2))
print('uncapped bat:',B[~B.capped].name.tolist()[:40]); print('uncapped bowl:',W[~W.capped].name.tolist()[:40])
print(B.position.value_counts().to_dict()); print(W.primary_phase.value_counts().to_dict())
print(B.country.value_counts().head(20).to_dict())
print('sample',out['sample']['match'],out['sample']['result'], len(out['sample']['balls']))
import os; print(os.path.getsize('/home/claude/dash/data.json'))

# ---------------- cap corrections (international caps in any format / before 2022, not visible in this T20I data) ----
CAPPED_MANUAL = {'Mahendra Singh Dhoni':'India','Ajinkya Rahane':'India','Krunal Pandya':'India','Nitish Rana':'India',
 'Manish Pandey':'India','Vijay Shankar':'India','Shahbaz Ahmed':'India','Rajat Patidar':'India','Devdutt Padikkal':'India',
 'Sunil Narine':'West Indies','Dwayne Bravo':'West Indies','T Natarajan':'India','Ishant Sharma':'India','Kuldeep Sen':'India',
 'Sai Kishore':'India','Sandeep Sharma':'India','Piyush Chawla':'India','Jaydev Unadkat':'India','Mohit Sharma':'India',
 'Anshul Kamboj':'India','Rahul Chahar':'India','Akash Deep':'India'}
NOTES = {'Mahendra Singh Dhoni':'Retired from internationals in 2019; IPL rules treat players 5+ years out of international cricket as uncapped.'}
for role in ('batters','bowlers'):
    for x in out[role]:
        if x['capped']:
            x['cap_src']='data'
        elif x['name'] in CAPPED_MANUAL:
            x.update(capped=True, country=CAPPED_MANUAL[x['name']], cap_src='manual')
        else:
            x.update(country='India', cap_src='inferred')
        x['note']=NOTES.get(x['name'],'')
# sample match batting sides
bb=pd.read_csv(D+'ball_by_ball/ball_by_ball.csv',dtype=str,on_bad_lines='skip',usecols=['match_id','innings','batting_team_id','is_wicket'])
bb=bb[bb.match_id==str(sm)]
side={int(i):teams.get(int(t),'') for i,t in bb.groupby('innings').batting_team_id.first().items() if str(i) in ('1','2')}
out['sample']['bat1']=side.get(1,''); out['sample']['bat2']=side.get(2,'')
# trim fields
KB=['player_id','name','rank','reliability','level','level_lo90','level_hi90','p_above_avg','level_base','clutch_delta','impact',
    'consistency','breadth','n','n_inn','p_top10','rank_p05','rank_p95','runs_per100','bnd_pct','avg_LI','seasons','wk_delta100',
    'country','capped','caps','cap_src','note','ipl_team','ipl_season','ipl_matches']
for role,extra in (('batters',['position','bat_pos','hand','level_chasing','level_setting','level_collapse_entry','level_death','level_vs_spin','level_vs_pace']),
                   ('bowlers',['primary_phase','phase_tags','share_pp','share_middle','share_death','bowl_type','pace_spin','level_defending',
                               'level_bowling_first','level_powerplay','level_middle','level_death','level_vs_set_batter'])):
    out[role]=[{k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items() if k in KB+extra} for x in out[role]]
json.dump(out,open('/home/claude/dash/data.json','w'),separators=(',',':'))
print('final size',os.path.getsize('/home/claude/dash/data.json'), out['sample']['bat1'], out['sample']['bat2'])
print(sum(x['cap_src']=='manual' for r in ('batters','bowlers') for x in out[r]), 'manual caps')
