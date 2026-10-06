import os, json, statistics
from datetime import datetime, timezone
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='飲食店メニュー検索 API', version='14.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

DB_URL = os.getenv('DATABASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_ANALYSIS_MODEL', 'gpt-6-luna')


def conn():
    if not DB_URL:
        raise HTTPException(503, 'DATABASE_URL が設定されていません')
    return psycopg.connect(DB_URL, row_factory=dict_row)


def parse_ingredients(value: str):
    parts = [x.strip() for x in (value or '').replace('、', ',').split(',') if x.strip()]
    # preserve order / remove duplicates
    return list(dict.fromkeys(parts))


def resolve_ids(cur, names):
    if not names:
        return []
    rows = cur.execute('''
      SELECT i.id, i.canonical_name, i.canonical_name AS matched_name
      FROM ingredients i
      WHERE i.canonical_name = ANY(%s)
      UNION ALL
      SELECT i.id, i.canonical_name, a.alias AS matched_name
      FROM ingredients i
      JOIN ingredient_aliases a ON a.ingredient_id=i.id
      WHERE a.alias = ANY(%s)
    ''', (names, names)).fetchall()
    mapping = {r['matched_name']: (r['id'], r['canonical_name']) for r in rows}
    resolved = []
    for n in names:
        if n in mapping:
            resolved.append((n, mapping[n][0]))
    return resolved


def common_where(prefecture=None, cuisine=None, min_price=None, max_price=None):
    clauses = ['m.status = \'active\'', "mi.review_status = 'approved'"]
    params = []
    if prefecture:
        clauses.append('r.prefecture = %s'); params.append(prefecture)
    if cuisine:
        clauses.append('r.cuisine_type = %s'); params.append(cuisine)
    if min_price is not None:
        clauses.append('m.price >= %s'); params.append(min_price)
    if max_price is not None:
        clauses.append('m.price <= %s'); params.append(max_price)
    return ' AND '.join(clauses), params


@app.get('/api/health')
def health():
    if not DB_URL:
        return {'ok': False, 'database': False, 'error': 'DATABASE_URL missing'}
    try:
        with conn() as c:
            c.execute('SELECT 1').fetchone()
        return {'ok': True, 'database': True, 'version': '14.0'}
    except Exception as e:
        return {'ok': False, 'database': False, 'error': str(e)[:180]}


def base_menu_query(cur, ingredient_names, prefecture=None, cuisine=None, min_price=None, max_price=None):
    resolved = resolve_ids(cur, ingredient_names)
    if len(resolved) != len(ingredient_names):
        missing = [n for n in ingredient_names if n not in {x[0] for x in resolved}]
        return [], resolved, missing
    ids = [x[1] for x in resolved]
    where, extra = common_where(prefecture, cuisine, min_price, max_price)
    sql = f'''
      SELECT m.id, m.name, m.description, m.price, m.style, m.source_url,
             m.last_verified_at, r.name AS restaurant_name, r.prefecture,
             r.city, r.cuisine_type,
             AVG(mi.confidence) AS confidence,
             COUNT(DISTINCT mi.ingredient_id) FILTER (WHERE mi.ingredient_id = ANY(%s::uuid[])) AS matched_count
      FROM menus m
      JOIN restaurants r ON r.id=m.restaurant_id
      JOIN menu_ingredients mi ON mi.menu_id=m.id
      WHERE {where}
      GROUP BY m.id, r.id
      HAVING COUNT(DISTINCT mi.ingredient_id) FILTER (WHERE mi.ingredient_id = ANY(%s::uuid[])) = %s
    '''
    rows = cur.execute(sql, (ids, *extra, ids, len(ids))).fetchall()
    return rows, resolved, []


@app.get('/api/search')
def search(ingredient: str, prefecture: Optional[str]=None, cuisine: Optional[str]=None,
          min_price: Optional[int]=None, max_price: Optional[int]=None, limit: int=20):
    names = parse_ingredients(ingredient)
    if not names:
        raise HTTPException(400, 'ingredient を指定してください')
    limit = max(1, min(limit, 50))
    try:
        with conn() as c:
            rows, resolved, missing = base_menu_query(c, names, prefecture, cuisine, min_price, max_price)
            if missing:
                return {'ingredients': names, 'resolved': [x[0] for x in resolved], 'missing': missing,
                        'total': 0, 'menus': []}
            # scoring: ingredient confidence + freshness + source presence + style/restaurant diversity
            now = datetime.now(timezone.utc)
            for r in rows:
                days = max(0, (now - r['last_verified_at']).days) if r['last_verified_at'] else 365
                freshness = max(0, 1 - min(days, 365)/365)
                r['_score'] = round(float(r['confidence'] or 0)*0.65 + freshness*0.20 + (0.15 if r['source_url'] else 0), 4)
            rows.sort(key=lambda x: x['_score'], reverse=True)
            # light diversity: don't show >3 items from same restaurant in the first page
            selected=[]; counts={}
            for r in rows:
                n=counts.get(r['restaurant_name'],0)
                if n>=3 and len(selected)<limit:
                    continue
                selected.append(r); counts[r['restaurant_name']]=n+1
                if len(selected)>=limit: break
            for r in selected: r.pop('_score', None)
            return {'ingredients': names, 'resolved': [x[0] for x in resolved], 'missing': [],
                    'total': len(rows), 'menus': selected}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f'search error: {str(e)[:220]}')


@app.get('/api/market')
def market(ingredient: str):
    names = parse_ingredients(ingredient)
    if not names: raise HTTPException(400, 'ingredient を指定してください')
    try:
        with conn() as c:
            rows, resolved, missing = base_menu_query(c, names)
            if missing:
                return {'ingredients': names, 'missing': missing, 'menu_count': 0, 'average_price': None,
                        'median_price': None, 'price_min': None, 'price_max': None,
                        'genres': [], 'regions': [], 'styles': [], 'cooccurring': [], 'whitespace': []}
            menu_ids=[r['id'] for r in rows]
            prices=[r['price'] for r in rows if r['price'] is not None]
            avg=round(statistics.mean(prices)) if prices else None
            median=statistics.median(prices) if prices else None
            median=int(median) if median is not None and float(median).is_integer() else median
            # distribution helpers based on matched menu IDs
            def dist(sql):
                return c.execute(sql, (menu_ids,)).fetchall() if menu_ids else []
            genres=dist('''SELECT r.cuisine_type AS name, COUNT(*) AS count FROM menus m JOIN restaurants r ON r.id=m.restaurant_id WHERE m.id=ANY(%s::uuid[]) GROUP BY r.cuisine_type ORDER BY count DESC''')
            regions=dist('''SELECT r.prefecture AS name, COUNT(*) AS count FROM menus m JOIN restaurants r ON r.id=m.restaurant_id WHERE m.id=ANY(%s::uuid[]) GROUP BY r.prefecture ORDER BY count DESC''')
            styles=dist('''SELECT COALESCE(style,'未分類') AS name, COUNT(*) AS count FROM menus WHERE id=ANY(%s::uuid[]) GROUP BY style ORDER BY count DESC''')
            requested_ids=[x[1] for x in resolved]
            co=c.execute('''
              SELECT i.canonical_name AS name, COUNT(DISTINCT mi.menu_id) AS count
              FROM menu_ingredients mi JOIN ingredients i ON i.id=mi.ingredient_id
              WHERE mi.menu_id=ANY(%s::uuid[]) AND mi.review_status='approved'
                AND NOT (i.id=ANY(%s::uuid[]))
              GROUP BY i.canonical_name ORDER BY count DESC, name LIMIT 15
            ''',(menu_ids, requested_ids)).fetchall()
            # whitespace: low-count cuisine/style combinations, explicitly a signal only
            whitespace=c.execute('''
              SELECT r.cuisine_type AS cuisine, COALESCE(m.style,'未分類') AS style, COUNT(*) AS count
              FROM menus m JOIN restaurants r ON r.id=m.restaurant_id
              WHERE m.id=ANY(%s::uuid[])
              GROUP BY r.cuisine_type, m.style ORDER BY count ASC, cuisine, style LIMIT 10
            ''',(menu_ids,)).fetchall()
            return {
              'ingredients': names, 'resolved': [x[0] for x in resolved], 'missing': [],
              'menu_count': len(rows), 'average_price': avg, 'median_price': median,
              'price_min': min(prices) if prices else None, 'price_max': max(prices) if prices else None,
              'genres': genres, 'regions': regions, 'styles': styles, 'cooccurring': co,
              'whitespace': whitespace,
              'whitespace_note': '件数が少ない組み合わせを探索シグナルとして表示しています。需要や成功を意味するものではありません。'
            }
    except Exception as e:
        raise HTTPException(500, f'market error: {str(e)[:220]}')


def ai_request(kind, facts, menus):
    if not OPENAI_API_KEY:
        return {'enabled': False, 'message': 'OPENAI_API_KEY が未設定です', 'facts': facts}
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)
    if kind=='analysis':
        instruction='''あなたは飲食店の商品開発向け市場アナリストです。与えられたDB事実だけを根拠に、日本語で簡潔に市場分析してください。数字、価格、件数、URL、店舗名を新しく推測・生成してはいけません。事実と解釈を分け、最後に商品開発上の示唆を3点ください。'''
    else:
        instruction='''あなたは飲食店の商品開発担当です。与えられたDB事実だけを根拠に、新商品案を5件作ってください。各案は「商品名・コンセプト・想定価格帯・ターゲット・差別化・必要食材・市場根拠」を含めてください。価格帯や件数を事実のように捏造せず、想定価格であることを明記してください。'''
    payload=json.dumps({'facts':facts,'representative_menus':menus},ensure_ascii=False)
    response=client.responses.create(model=OPENAI_MODEL, input=[{'role':'system','content':instruction},{'role':'user','content':payload}])
    return {'enabled':True,'model':OPENAI_MODEL,'text':response.output_text}


@app.get('/api/ai/analysis')
def ai_analysis(ingredient: str):
    facts=market(ingredient)
    result=search(ingredient,limit=20)
    return ai_request('analysis',facts,result['menus'])

@app.get('/api/ai/ideas')
def ai_ideas(ingredient: str):
    facts=market(ingredient)
    result=search(ingredient,limit=20)
    return ai_request('ideas',facts,result['menus'])
