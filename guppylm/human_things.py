"""Human things the fish doesn't understand, for the "confused" topic.

With 31 nouns, "repeat the user's word" and "say one of these 31 words" look the
same to the model. A list this long means most nouns appear once or not at all
in a 60K-sample dataset, so naming the thing requires copying it from the prompt.

Keep tools/memorization.py's COPY_THINGS (and close variants) out of this list,
or the copy probe stops measuring anything.
"""

# Countable nouns, singular, no article. An article is added at use.
_COUNTABLE = """
phone laptop tablet keyboard mouse monitor printer scanner router modem charger
cable headphone earbud speaker microphone webcam smartwatch remote controller
television radio stereo turntable record cassette camera tripod projector
calculator clock watch wallet purse backpack suitcase briefcase umbrella
raincoat jacket sweater hoodie scarf glove mitten hat cap helmet sock shoe
sneaker boot sandal slipper belt tie necklace bracelet ring earring sunglasses
mirror comb hairbrush toothbrush razor towel blanket pillow mattress couch sofa
armchair recliner desk chair stool bench table bookshelf dresser wardrobe closet
drawer cabinet lamp chandelier candle fireplace radiator thermostat ceiling fan
air conditioner heater dehumidifier humidifier doorbell mailbox fence gate
driveway garage shed porch balcony attic basement staircase elevator escalator
hallway kitchen bathroom bedroom office cubicle warehouse factory skyscraper
apartment condo townhouse mansion cabin tent hotel motel hostel airport
train station bus stop subway platform parking lot gas station car wash
laundromat bakery pharmacy hospital clinic dentist library museum theater
cinema stadium arena gym spa salon barbershop restaurant diner cafe bar pub
nightclub casino bank post office courthouse city hall prison school bus
truck van motorcycle scooter skateboard tricycle unicycle wagon stroller
wheelbarrow lawnmower leaf blower snowblower chainsaw hammer screwdriver wrench
drill saw ladder shovel rake hose sprinkler wheel tire engine steering wheel
seatbelt windshield license plate traffic light stop sign crosswalk sidewalk
highway freeway bridge tunnel roundabout intersection map compass globe
calendar notebook diary journal pen pencil marker crayon eraser stapler
paperclip envelope stamp postcard letter package box crate bag basket
shopping cart receipt coupon gift card voucher ticket invoice paycheck salary
loan mortgage budget bill deposit refund pension retirement plan lottery ticket
contract lease warranty insurance policy subscription membership newsletter
magazine newspaper novel textbook dictionary encyclopedia comic book cookbook
brochure flyer poster billboard advertisement commercial trailer sequel
documentary sitcom soap opera reality show game show talk show cartoon anime
video game board game card game puzzle crossword sudoku chessboard dice
deck of cards jigsaw puzzle kite yoyo frisbee trampoline swing slide seesaw
sandbox playground carousel roller coaster ferris wheel water park zoo circus
parade festival concert opera ballet musical play rehearsal audition recital
guitar piano violin cello drum trumpet saxophone flute clarinet harmonica
banjo ukulele accordion tuba harp synthesizer metronome song album playlist
ringtone jingle anthem lullaby karaoke machine jukebox microwave oven stove
refrigerator freezer dishwasher blender mixer kettle coffee maker teapot
frying pan saucepan spatula whisk ladle colander cutting board knife fork
spoon chopstick plate bowl mug cup glass straw napkin tablecloth placemat
lunchbox thermos water bottle ice cube tray grill barbecue picnic sandwich
burger hot dog pizza taco burrito salad soup steak omelet pancake waffle
donut bagel muffin cupcake cookie brownie pie cake sundae milkshake smoothie
latte cappuccino espresso soda lemonade cocktail popsicle candy bar lollipop
pretzel cracker potato chip french fry onion ring nugget casserole lasagna
spaghetti noodle dumpling sushi roll burrito bowl birthday party wedding
funeral graduation anniversary reunion holiday vacation honeymoon road trip
cruise flight layover boarding pass visa itinerary souvenir
job interview resume cover letter promotion raise deadline spreadsheet cell
presentation slideshow memo report agenda conference webinar workshop seminar
lecture exam quiz essay thesis diploma degree scholarship tuition loan
classroom chalkboard whiteboard backpack strap locker cafeteria principal
teacher student professor tutor coach referee umpire lawyer judge jury
senator mayor governor president election ballot campaign debate protest
petition lawsuit trial verdict sentence fine parking ticket speeding ticket
traffic jam commute carpool rideshare taxi limousine ambulance fire truck
police car helicopter airplane jet rocket satellite space station telescope
microscope laboratory experiment hypothesis theory equation formula graph
chart diagram algorithm database server website webpage browser search engine
hashtag emoji meme selfie screenshot username profile avatar inbox
notification reminder alarm timer stopwatch countdown app store update
download upload backup cloud drive hard drive flash drive memory card battery
power outlet extension cord light switch light bulb fuse box circuit breaker
generator solar panel wind turbine power plant pipeline oil rig coal mine
skyscraper crane bulldozer forklift tractor combine harvester scarecrow barn
silo farmhouse vineyard orchard greenhouse garden hose flowerpot doormat
welcome mat key keychain lock padlock safe vault alarm system security camera
smoke detector fire extinguisher first aid kit bandage thermometer stethoscope
wheelchair crutch cast pill vitamin prescription vaccine syringe x-ray
appointment checkup surgery diagnosis therapist dentist chair braces retainer
hearing aid contact lens eyeglass case perfume cologne deodorant shampoo
conditioner lotion sunscreen lipstick nail polish makeup bag hair dryer
curling iron straightener tattoo piercing haircut wig mustache beard
handshake hug high five wink sneeze hiccup yawn sigh grudge rumor gossip
secret promise apology compliment insult excuse alibi lie joke riddle pun
prank surprise party gift wrap bow ribbon greeting card wedding ring tuxedo
gown suit uniform costume mask pajama robe bathrobe swimsuit bikini wetsuit
snorkel surfboard paddleboard kayak canoe sailboat yacht jet ski motorboat
ferry cruise ship submarine lighthouse pier boardwalk beach towel beach ball
sandcastle sunburn snowman snowball sled ski snowboard ice skate hockey stick
golf club tennis racket baseball bat football soccer ball volleyball net
bowling ball dartboard pool table ping pong paddle treadmill dumbbell
barbell yoga mat jump rope stopwatch scoreboard trophy medal ribbon jersey
mascot cheerleader stadium seat hot tub sauna steam room swimming pool diving
board lifeguard chair hammock lawn chair patio umbrella fire pit bonfire
campfire marshmallow sleeping bag flashlight lantern backpacking stove hiking
trail mountain bike map app gps tracker step counter fitness tracker
smart speaker voice assistant robot vacuum drone hoverboard segway electric car
charging station parking meter toll booth speed bump pothole detour shortcut
""".split("\n")

# Mass nouns, activities, and named things. Used as written.
_BARE = """
politics, money, the internet, email, taxes, driving, school, work, math,
reading, the news, social media, cooking, shopping, rent, the stock market,
homework, wifi, bluetooth, cryptocurrency, electricity, plumbing, gasoline,
traffic, insurance, healthcare, retirement, inflation, interest rates,
debt, savings, paperwork, bureaucracy, customer service, tech support, voicemail,
texting, online dating, small talk, networking, marketing, advertising, branding,
accounting, bookkeeping, payroll, overtime, vacation days, sick leave, office
politics, the weekend, monday mornings, rush hour, jet lag, daylight saving time,
time zones, calendars, deadlines, meetings, emails, passwords, usernames,
spam, malware, viruses, hackers, firewalls, encryption, coding, programming,
javascript, python, html, spreadsheets, databases, artificial intelligence,
machine learning, robots, algorithms, blockchain, the cloud, streaming, binge
watching, podcasts, audiobooks, video calls, group chats, memes, emojis,
hashtags, influencers, followers, likes, comments, going viral, clickbait,
online shopping, free shipping, returns, gift wrapping, black friday, tipping,
sales tax, coupons, loyalty points, credit scores, mortgages, property taxes,
home repairs, yard work, mowing the lawn, raking leaves, shoveling snow, laundry,
ironing, folding clothes, vacuuming, dusting, mopping, doing the dishes, grocery
shopping, meal prep, dieting, calories, carbs, protein shakes, intermittent
fasting, veganism, gluten, caffeine, coffee, tea, wine, beer, chocolate, bread,
cheese, butter, peanut butter, ketchup, mustard, mayonnaise, hot sauce, salt,
pepper, sugar, flour, rice, pasta, cereal, oatmeal, yogurt, ice cream,
popcorn, takeout, fast food, food delivery, brunch, happy hour, dinner parties,
potlucks, barbecues, picnics, camping, hiking, jogging, running, cycling,
weightlifting, yoga, pilates, crossfit, marathons, stretching, golf, tennis,
baseball, football, soccer, hockey, bowling, skiing, snowboarding, surfing,
skateboarding, rock climbing, bungee jumping, skydiving, fishing, hunting,
gardening, knitting, sewing, woodworking, painting, drawing, sculpture,
photography, filmmaking, dancing, singing, acting, stand up comedy, magic tricks,
juggling, chess, poker, bingo, gambling, the lottery, horoscopes, astrology,
tarot, philosophy, psychology, sociology, economics, history, geography,
chemistry, biology, geology, astronomy, engineering, architecture, law, medicine,
nursing, dentistry, accounting exams, grammar, spelling, punctuation, handwriting,
cursive, typing, public speaking, job hunting, commuting, carpooling, parking,
speeding, road rage, gps, directions, airports, security lines, luggage, carry
ons, layovers, customs, jet skis, cruises, road trips, hotels, room service,
tourism, sightseeing, museums, art galleries, fashion, makeup, skincare,
haircuts, hair dye, shaving, perfume, jewelry, high heels, neckties, suits,
pajamas, laundry day, dry cleaning, retail, wholesale, real estate, landlords,
roommates, neighbors, in laws, babysitting, daycare, homeschooling, report cards,
parent teacher conferences, college, student loans, graduation, retirement
parties, birthdays, anniversaries, weddings, divorce, therapy, meditation,
mindfulness, self care, sleep schedules, insomnia, alarm clocks, snooze buttons,
naps, caffeine crashes, hangovers, allergies, the flu, colds, headaches,
backaches, doctor visits, dentist visits, flossing, vaccines, health insurance,
gym memberships, new year's resolutions, valentine's day, halloween,
thanksgiving, christmas shopping, fireworks, the fourth of july, elections,
voting, democracy, congress, the senate, taxes due, jury duty, the dmv,
paperwork, notary stamps, small claims court, parking tickets, speeding
tickets, the post office, junk mail, package tracking, porch pirates, smart
homes, home security, garage door openers, sprinkler systems, pest control,
termites, mold, leaky faucets, clogged drains, power outages, the electric
bill, the water bill, the gas bill, recycling, composting, landfills, pollution,
climate change, carbon footprints, electric cars, gas prices, oil changes,
car insurance, car payments, road trips, traffic cameras, toll roads, cruise
control, parallel parking, snow tires, windshield wipers
""".replace("\n", " ").split(",")


def _with_article(noun):
    return ("an " if noun[0] in "aeiou" else "a ") + noun


# The original 31, kept first and verbatim: the copy probe's KNOWN_THINGS control uses them.
_ORIGINAL = [
    "politics", "money", "the internet", "email", "taxes", "a phone",
    "driving", "a movie", "school", "work", "a computer", "math",
    "reading", "the news", "social media", "cooking", "shopping",
    "a job", "rent", "the stock market", "a meeting", "homework",
    "an app", "a password", "wifi", "bluetooth", "a podcast",
    "cryptocurrency", "a spreadsheet", "an alarm clock", "a car",
]


def _build():
    countable = [w for line in _COUNTABLE for w in _split_line(line)]
    things = _ORIGINAL + [_with_article(n) for n in countable] + [b.strip() for b in _BARE]
    seen, out = set(), []
    for t in things:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# Multi-word countables, matched greedily before splitting the rest on spaces.
_COMPOUNDS = {c for c in """
ceiling fan|air conditioner|train station|bus stop|subway platform|parking lot
gas station|car wash|post office|city hall|school bus|leaf blower|steering wheel
license plate|traffic light|stop sign|shopping cart|gift card|lottery ticket
insurance policy|comic book|soap opera|reality show|game show|talk show
video game|board game|card game|deck of cards|jigsaw puzzle|roller coaster
ferris wheel|water park|karaoke machine|coffee maker|frying pan|cutting board
lunchbox|water bottle|ice cube tray|hot dog|candy bar|potato chip|french fry
onion ring|sushi roll|burrito bowl|birthday party|road trip|boarding pass
job interview|cover letter|traffic jam|parking ticket|speeding ticket
fire truck|police car|space station|search engine|app store|hard drive
flash drive|memory card|power outlet|extension cord|light switch|light bulb
fuse box|circuit breaker|solar panel|wind turbine|power plant|oil rig
coal mine|combine harvester|garden hose|welcome mat|alarm system
security camera|smoke detector|fire extinguisher|first aid kit|hearing aid
contact lens|eyeglass case|nail polish|makeup bag|hair dryer|curling iron
high five|surprise party|gift wrap|greeting card|wedding ring|cruise ship
jet ski|beach towel|beach ball|ice skate|hockey stick|golf club
tennis racket|baseball bat|soccer ball|bowling ball|pool table
ping pong paddle|yoga mat|jump rope|stadium seat|hot tub|steam room
swimming pool|diving board|lifeguard chair|lawn chair|patio umbrella|fire pit
sleeping bag|backpacking stove|hiking trail|mountain bike|map app|gps tracker
step counter|fitness tracker|smart speaker|voice assistant|robot vacuum
electric car|charging station|parking meter|toll booth|speed bump|bus stop
cell|cover letter|school bus|retirement plan|dentist chair|lawn chair
""".replace("\n", "|").split("|") if c}


def _split_line(line):
    words, out, i = line.split(), [], 0
    while i < len(words):
        for n in (4, 3, 2):
            phrase = " ".join(words[i:i + n])
            if len(words[i:i + n]) == n and phrase in _COMPOUNDS:
                out.append(phrase)
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


HUMAN_THINGS = _build()
