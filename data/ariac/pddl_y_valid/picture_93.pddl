(define (problem picture_93)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_battery red_regulator red_pump_1 - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear green_battery)
        (clear red_pump_1)
        (part_at red_pump_1 table)
        (part_at red_pump pump_placement)
        (part_at green_battery battery_placement)
        (part_at red_regulator regulator_placement)
        (clear red_regulator)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
