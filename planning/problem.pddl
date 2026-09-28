(define (problem cifar100-full-setup)
  (:domain cifar100-process)

  (:objects
    ;; --------------------------------------------------------------------
    ;; THE 100 CIFAR OBJECTS (all type: item)
    ;; --------------------------------------------------------------------
    apple aquarium-fish baby bear beaver bed bee beetle bicycle bottle
    bowl boy bridge bus butterfly camel can castle caterpillar cattle
    chair chimpanzee clock cloud cockroach couch crab crocodile cup
    dinosaur dolphin elephant flatfish forest fox girl hamster house
    kangaroo keyboard lamp lawn-mower leopard lion lizard lobster man
    maple motorcycle mountain mouse mushroom oak orange orchid otter
    palm pear pickup-truck pine plain plate poppy porcupine possum
    rabbit raccoon ray road rocket rose sea seal shark shrew skunk
    skyscraper snail snake spider squirrel streetcar sunflower
    sweet-pepper table tank telephone television tiger tractor train
    trout tulip turtle wardrobe whale willow wolf woman worm
    - item
  )

  (:init
    ;; --------------------------------------------------------------------
    ;; AGENT INITIAL STATE
    ;; --------------------------------------------------------------------
    (hand-empty)
    (agent-at lab)

    ;; --------------------------------------------------------------------
    ;; TOOLS (always at lab, always clear, always marked as tools)
    ;; --------------------------------------------------------------------
    (at knife lab)
    (at dslr lab)
    (clear knife)
    (clear dslr)
    (is-tool knife)
    (is-tool dslr)

    ;; --------------------------------------------------------------------
    ;; SURFACES (objects that can have things stacked on them)
    ;; --------------------------------------------------------------------
    (is-surface table)
    (is-surface bed)
    (is-surface chair)
    (is-surface couch)
    (is-surface road)
    (is-surface bridge)
    (is-surface plain)
    (is-surface sea)
    (is-surface plate)
    (is-surface bowl)

    ;; --------------------------------------------------------------------
    ;; LAB OBJECTS - FURNITURE (on floor, clear)
    ;; --------------------------------------------------------------------
    (at table lab) (whole table) (clear table)
    (at bed lab) (whole bed) (clear bed)
    (at chair lab) (whole chair) (clear chair)
    (at couch lab) (whole couch) (clear couch)
    (at wardrobe lab) (whole wardrobe) (clear wardrobe)

    ;; --------------------------------------------------------------------
    ;; LAB OBJECTS - ON TABLE (stacked on table)
    ;; --------------------------------------------------------------------

    ;; Tools on table
    (on-top knife table)
    (on-top dslr table)

    ;; Small electronics on table
    (at clock lab) (whole clock) (clear clock) (on-top clock table)
    (at keyboard lab) (whole keyboard) (clear keyboard) (on-top keyboard table)
    (at telephone lab) (whole telephone) (clear telephone) (on-top telephone table)
    (at lamp lab) (whole lamp) (clear lamp) (on-top lamp table)

    ;; Containers on table
    (at bottle lab) (whole bottle) (clear bottle) (on-top bottle table)
    (at bowl lab) (whole bowl) (clear bowl) (on-top bowl table)
    (at can lab) (whole can) (clear can) (on-top can table)
    (at cup lab) (whole cup) (clear cup) (on-top cup table)
    (at plate lab) (whole plate) (clear plate) (on-top plate table)

    ;; Food on table
    (at apple lab) (whole apple) (clear apple) (on-top apple table)
    (at mushroom lab) (whole mushroom) (clear mushroom) (on-top mushroom table)
    (at orange lab) (whole orange) (clear orange) (on-top orange table)
    (at pear lab) (whole pear) (clear pear) (on-top pear table)
    (at sweet-pepper lab) (whole sweet-pepper) (clear sweet-pepper) (on-top sweet-pepper table)

    ;; Small animals on table
    (at aquarium-fish lab) (whole aquarium-fish) (clear aquarium-fish) (on-top aquarium-fish table)
    (at hamster lab) (whole hamster) (clear hamster) (on-top hamster table)
    (at mouse lab) (whole mouse) (clear mouse) (on-top mouse table)
    (at snail lab) (whole snail) (clear snail) (on-top snail table)
    (at spider lab) (whole spider) (clear spider) (on-top spider table)
    (at worm lab) (whole worm) (clear worm) (on-top worm table)
    (at cockroach lab) (whole cockroach) (clear cockroach) (on-top cockroach table)

    ;; --------------------------------------------------------------------
    ;; LAB OBJECTS - ON FLOOR (not on table, clear)
    ;; --------------------------------------------------------------------
    (at television lab) (whole television) (clear television)
    (at baby lab) (whole baby) (clear baby)
    (at boy lab) (whole boy) (clear boy)
    (at girl lab) (whole girl) (clear girl)
    (at man lab) (whole man) (clear man)
    (at woman lab) (whole woman) (clear woman)

    ;; --------------------------------------------------------------------
    ;; OUTDOOR OBJECTS - ALL CLEAR AND ON GROUND
    ;; --------------------------------------------------------------------

    ;; Flowers
    (at orchid outdoors) (whole orchid) (clear orchid)
    (at rose outdoors) (whole rose) (clear rose)
    (at tulip outdoors) (whole tulip) (clear tulip)
    (at sunflower outdoors) (whole sunflower) (clear sunflower)
    (at poppy outdoors) (whole poppy) (clear poppy)

    ;; Structures
    (at bridge outdoors) (whole bridge) (clear bridge)
    (at castle outdoors) (whole castle) (clear castle)
    (at house outdoors) (whole house) (clear house)
    (at road outdoors) (whole road) (clear road)
    (at skyscraper outdoors) (whole skyscraper) (clear skyscraper)

    ;; Natural features
    (at cloud outdoors) (whole cloud) (clear cloud)
    (at forest outdoors) (whole forest) (clear forest)
    (at mountain outdoors) (whole mountain) (clear mountain)
    (at plain outdoors) (whole plain) (clear plain)
    (at sea outdoors) (whole sea) (clear sea)

    ;; Trees
    (at maple outdoors) (whole maple) (clear maple)
    (at oak outdoors) (whole oak) (clear oak)
    (at palm outdoors) (whole palm) (clear palm)
    (at pine outdoors) (whole pine) (clear pine)
    (at willow outdoors) (whole willow) (clear willow)

    ;; Vehicles
    (at bicycle outdoors) (whole bicycle) (clear bicycle)
    (at bus outdoors) (whole bus) (clear bus)
    (at motorcycle outdoors) (whole motorcycle) (clear motorcycle)
    (at pickup-truck outdoors) (whole pickup-truck) (clear pickup-truck)
    (at train outdoors) (whole train) (clear train)
    (at lawn-mower outdoors) (whole lawn-mower) (clear lawn-mower)
    (at rocket outdoors) (whole rocket) (clear rocket)
    (at streetcar outdoors) (whole streetcar) (clear streetcar)
    (at tank outdoors) (whole tank) (clear tank)
    (at tractor outdoors) (whole tractor) (clear tractor)

    ;; Large mammals
    (at bear outdoors) (whole bear) (clear bear)
    (at leopard outdoors) (whole leopard) (clear leopard)
    (at lion outdoors) (whole lion) (clear lion)
    (at tiger outdoors) (whole tiger) (clear tiger)
    (at wolf outdoors) (whole wolf) (clear wolf)
    (at camel outdoors) (whole camel) (clear camel)
    (at cattle outdoors) (whole cattle) (clear cattle)
    (at chimpanzee outdoors) (whole chimpanzee) (clear chimpanzee)
    (at elephant outdoors) (whole elephant) (clear elephant)
    (at kangaroo outdoors) (whole kangaroo) (clear kangaroo)

    ;; Aquatic & small animals
    (at beaver outdoors) (whole beaver) (clear beaver)
    (at dolphin outdoors) (whole dolphin) (clear dolphin)
    (at otter outdoors) (whole otter) (clear otter)
    (at seal outdoors) (whole seal) (clear seal)
    (at whale outdoors) (whole whale) (clear whale)
    (at flatfish outdoors) (whole flatfish) (clear flatfish)
    (at ray outdoors) (whole ray) (clear ray)
    (at shark outdoors) (whole shark) (clear shark)
    (at trout outdoors) (whole trout) (clear trout)
    (at crocodile outdoors) (whole crocodile) (clear crocodile)
    (at dinosaur outdoors) (whole dinosaur) (clear dinosaur)
    (at fox outdoors) (whole fox) (clear fox)
    (at porcupine outdoors) (whole porcupine) (clear porcupine)
    (at possum outdoors) (whole possum) (clear possum)
    (at raccoon outdoors) (whole raccoon) (clear raccoon)
    (at skunk outdoors) (whole skunk) (clear skunk)

    ;; Crustaceans & insects
    (at crab outdoors) (whole crab) (clear crab)
    (at lobster outdoors) (whole lobster) (clear lobster)
    (at bee outdoors) (whole bee) (clear bee)
    (at beetle outdoors) (whole beetle) (clear beetle)
    (at butterfly outdoors) (whole butterfly) (clear butterfly)
    (at caterpillar outdoors) (whole caterpillar) (clear caterpillar)

    ;; Small outdoor animals
    (at rabbit outdoors) (whole rabbit) (clear rabbit)
    (at squirrel outdoors) (whole squirrel) (clear squirrel)
    (at shrew outdoors) (whole shrew) (clear shrew)
    (at lizard outdoors) (whole lizard) (clear lizard)
    (at snake outdoors) (whole snake) (clear snake)
    (at turtle outdoors) (whole turtle) (clear turtle)
  )
)
