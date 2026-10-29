"""Apply an explicit reviewed carry-forward selection; preserve original and lowercase views."""
import gzip,json,collections
from pathlib import Path
from tools.aquarium.v52.data import digest,file_hash
ROOT=Path('/private/tmp/guppylm-dev-apply');LIB=ROOT/'experiments/v52/training-library'
CHOICES={
 '01506c04a91b8cf0695cb37b33be95a531158ba28248592788f7f05ebf93b791':'Playful fish-scale airplane answer and eager riddle game.',
 '0161873844ce1a32ea543be23f6c6d3f89d132cef7cfdb4547c6c3835347ae7f':'Observational water colors and a charming silver-at-night farewell.',
 '20f3a5c2851f86b64b1ac00e677661cb78811b40586f232f3be2fc2526848b3c':'Warm companionship and even fast fish need a nap.',
 '0ebe50374c224baaaa7dd2d12b27759de67a83148eac567bf4f2d93a045cce59':'Responds to the soaked-book situation without demanding precision.',
 '11016f60ffcd0e423d1e81102d3fd5bb618e4ad5f94dfbdd83722f3ce6366d5f':'Requested sun-on-water poem; keep poetry when invited.',
 '1f0e8165d7e7043575e7f63791a1823c680c5ac3c2ceb7c1e811bb4175dbdbc6':'Requested penguin chant with musical underwater imagery.',
 '04aed48113bcc1e42a3a3287f786d6948efbd95a39ba6c9e791aff3666c2d46e':'Carrot baton is silly and understandable.',
 '00e998ecf8eb53879aa780c89282c0ab41a8db3a24879fe494535b624f16ff82':'Different beliefs need not interfere with making puppets together.',
 '0158e6798a30f5b39ad86a2b16a21d9f9dbb99f70ced0615a940dd55ca94ced6':'Learns a pretend word after the visitor supplies its meaning.',
 '017b7947875137ab70f6acfbfd06965fbc1fb174dc265df14a189a4545696734':'Small explicit mapping for a story, scoped to that story.',
 '0370b4a0d9f533c70857f3ec971a8daf9e4fc67652d03edcd1460e114fcb0c9b':'A brief relevant answer to a direct color question.',
 '0f5d1d620b3b87bb22deecac0e7cbcd87fef2fa84ef4d8cbc4dfa272cdfe6f8e':'Simple separate-turn recall of kindness as a pretend meaning.',
 '01093feec5d70e7d61ee2f332fe889104df79cdc9e4524ac40031a1c46950677':'Cloud library lends weather books with soft gray covers.',
 '00a8505c4671bf4f37b6384169923861dcc9a562524e9b4a06505472b2367798':'Enthusiastic pebble exchange, preserved from uppercase training.',
 '0b31614c33be1d755afa1f5792f3eeb9a239ef53bfdcc94331dcc207fe7ff49e':'Manatee friends, splashing and sea grass retain creature identity.',
}
def main():
 index=json.loads((LIB/'index.json').read_text());found={};members=collections.defaultdict(list)
 for h,info in index['datasets'].items():
  path=LIB/info['file'];assert file_hash(path)==info['file_sha256']
  for line in gzip.decompress(path.read_bytes()).decode().splitlines():
   row=json.loads(line)
   if row['uid'] in CHOICES:found[row['uid']]=row;members[row['uid']].append(h)
 assert found.keys()==CHOICES.keys()
 coverage={r['run']:[uid for uid,hs in members.items() if r['train_hash'] in hs] for r in index['runs']}
 assert all(coverage.values()),'Every run must contribute retained content'
 original=[found[uid] for uid in CHOICES];lower=[]
 for row in original:
  system=row['system'].lower();turns=[{**t,'content':t['content'].lower()} for t in row['turns']]
  lower.append({**row,'system':system,'turns':turns,'uid':digest([row['uid'],'lowercase-view-v1',system,turns]),'case_transform':{'from_uid':row['uid'],'rule':'lowercase system and all conversation content; originals retained'}})
 for name,rows in [('carry-forward-original.jsonl',original),('carry-forward-lowercase.jsonl',lower)]:
  path=LIB/name
  if path.exists():raise ValueError('Refusing existing carry-forward output')
  path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (LIB/'carry-forward-review.json').write_text(json.dumps({'purpose':'Reviewed seed set for future mixtures, not a complete training corpus or a finalized sampling weight. Original system restrictions are preserved in this historical selection; harmonize the persona system when building a new corpus.','decisions':[{'uid':uid,'reason':reason,'datasets':members[uid]} for uid,reason in CHOICES.items()],'run_coverage':coverage},indent=2)+'\n')
 print(len(original),'reviewed rows; all',len(coverage),'runs represented')
if __name__=='__main__':main()
